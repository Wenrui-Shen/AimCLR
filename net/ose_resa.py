import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchlight import import_class


def _build_projector(input_dim, hidden_dim, output_dim, num_layers):
    if num_layers < 2:
        raise ValueError('ReSA projector requires at least two layers')

    layers = []
    in_dim = input_dim
    for _ in range(num_layers - 1):
        layers.extend([
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        ])
        in_dim = hidden_dim
    layers.extend([
        nn.Linear(in_dim, output_dim),
        nn.BatchNorm1d(output_dim, affine=False),
    ])
    return nn.Sequential(*layers)


def _build_predictor(input_dim, hidden_dim):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, input_dim),
    )


def _build_instance_projector(input_dim, output_dim):
    """AimCLR-style MLP kept separate from the ReSA/OSE projector."""
    return nn.Sequential(
        nn.Linear(input_dim, input_dim),
        nn.ReLU(inplace=True),
        nn.Linear(input_dim, output_dim),
    )


class OSEResA(nn.Module):
    """ReSA with exemplar-guided prototype and interpolation losses."""

    def __init__(self, base_encoder=None, pretrain=True, feature_dim=256,
                 projector_hidden_dim=2048, projector_layers=3,
                 use_predictor=True, ose_enabled=True, queue_size=8192,
                 queue_contrast_enabled=False, instance_feature_dim=128,
                 instance_queue_size=32768, instance_temperature=0.07,
                 cluster_temperature=0.4, sinkhorn_temperature=0.05,
                 sinkhorn_iterations=3, in_channels=3, hidden_channels=16,
                 hidden_dim=256, num_class=60, dropout=0.5,
                 ose_prototype_stage=0,
                 graph_args={'layout': 'ntu-rgb+d', 'strategy': 'spatial'},
                 edge_importance_weighting=True, **kwargs):
        super().__init__()
        base_encoder = import_class(base_encoder)
        self.pretrain = pretrain
        self.ose_enabled = bool(ose_enabled)
        self.ose_prototype_stage = int(ose_prototype_stage)
        if self.ose_prototype_stage not in (0, 1, 2, 3):
            raise ValueError('ose_prototype_stage must be one of 0, 1, 2, 3')
        self.queue_contrast_enabled = bool(queue_contrast_enabled)
        if self.queue_contrast_enabled and not self.ose_enabled:
            raise ValueError(
                'Category-corrected queue contrast requires OSE')

        self.encoder_q = base_encoder(
            in_channels=in_channels, hidden_channels=hidden_channels,
            hidden_dim=hidden_dim, num_class=num_class, dropout=dropout,
            graph_args=graph_args,
            edge_importance_weighting=edge_importance_weighting, **kwargs)

        if not pretrain:
            return

        self.encoder_k = base_encoder(
            in_channels=in_channels, hidden_channels=hidden_channels,
            hidden_dim=hidden_dim, num_class=num_class, dropout=dropout,
            graph_args=graph_args,
            edge_importance_weighting=edge_importance_weighting, **kwargs)
        self.projector_q = _build_projector(
            hidden_dim, projector_hidden_dim, feature_dim, projector_layers)
        self.projector_k = _build_projector(
            hidden_dim, projector_hidden_dim, feature_dim, projector_layers)
        self.predictor = (_build_predictor(feature_dim, projector_hidden_dim)
                          if use_predictor else nn.Identity())

        self.cluster_temperature = float(cluster_temperature)
        self.sinkhorn_temperature = float(sinkhorn_temperature)
        self.sinkhorn_iterations = int(sinkhorn_iterations)
        if self.ose_enabled:
            self.queue_size = int(queue_size)
            self.register_buffer('queue', torch.zeros(feature_dim, queue_size))
            self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))
            self.register_buffer(
                'queue_filled', torch.zeros(1, dtype=torch.long))
            self.register_buffer(
                'queue_sample_indices',
                torch.full((queue_size,), -1, dtype=torch.long))
        if self.queue_contrast_enabled:
            if int(num_class) <= 0:
                raise ValueError('num_class must be positive')
            self.instance_feature_dim = int(instance_feature_dim)
            self.instance_queue_size = int(instance_queue_size)
            self.instance_temperature = float(instance_temperature)
            if self.instance_feature_dim <= 0:
                raise ValueError('instance_feature_dim must be positive')
            if self.instance_queue_size <= 0:
                raise ValueError('instance_queue_size must be positive')
            if self.instance_temperature <= 0:
                raise ValueError('instance_temperature must be positive')

            self.instance_projector_q = _build_instance_projector(
                hidden_dim, self.instance_feature_dim)
            self.instance_projector_k = _build_instance_projector(
                hidden_dim, self.instance_feature_dim)
            instance_queue = torch.randn(
                self.instance_feature_dim, self.instance_queue_size)
            self.register_buffer(
                'instance_queue', F.normalize(instance_queue, dim=0))
            self.register_buffer(
                'category_queue',
                torch.full(
                    (num_class, self.instance_queue_size),
                    1.0 / float(num_class)))
            self.register_buffer(
                'confidence_queue',
                torch.zeros(self.instance_queue_size))
            self.register_buffer(
                'instance_queue_ptr', torch.zeros(1, dtype=torch.long))
        self.reset_momentum_encoder()

    @torch.no_grad()
    def reset_momentum_encoder(self):
        if not self.pretrain:
            return
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        for param_q, param_k in zip(self.projector_q.parameters(),
                                    self.projector_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        if self.queue_contrast_enabled:
            for param_q, param_k in zip(
                    self.instance_projector_q.parameters(),
                    self.instance_projector_k.parameters()):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False

    @torch.no_grad()
    def _momentum_update(self, momentum):
        momentum = float(momentum)
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data.mul_(momentum).add_(param_q.data, alpha=1.0 - momentum)
        for param_q, param_k in zip(self.projector_q.parameters(),
                                    self.projector_k.parameters()):
            param_k.data.mul_(momentum).add_(param_q.data, alpha=1.0 - momentum)
        if self.queue_contrast_enabled:
            for param_q, param_k in zip(
                    self.instance_projector_q.parameters(),
                    self.instance_projector_k.parameters()):
                param_k.data.mul_(momentum).add_(
                    param_q.data, alpha=1.0 - momentum)

    @torch.no_grad()
    def _sinkhorn_knopp(self, scores):
        logits = scores / max(self.sinkhorn_temperature, 1e-12)
        logits = logits - logits.max()
        assignment = torch.exp(logits).t()
        assignment /= assignment.sum().clamp_min(1e-12)

        num_samples = assignment.shape[1]
        num_clusters = assignment.shape[0]
        for _ in range(self.sinkhorn_iterations):
            assignment /= assignment.sum(dim=1, keepdim=True).clamp_min(1e-12)
            assignment /= num_clusters
            assignment /= assignment.sum(dim=0, keepdim=True).clamp_min(1e-12)
            assignment /= num_samples
        assignment *= num_samples
        return assignment.t().detach()

    @staticmethod
    def _soft_cross_entropy(logits, target):
        return -(target * F.log_softmax(logits, dim=1)).sum(dim=1).mean()

    @staticmethod
    def _component_competition_scores(exemplar_z, components, class_index,
                                      alpha):
        similarity = torch.matmul(exemplar_z, components.t())
        class_similarity = similarity[class_index]
        if exemplar_z.size(0) > 1:
            other_mask = torch.ones(
                exemplar_z.size(0), dtype=torch.bool,
                device=exemplar_z.device)
            other_mask[class_index] = False
            max_other = similarity[other_mask].max(dim=0)[0]
        else:
            max_other = torch.zeros_like(class_similarity)
        return (
            float(alpha) * class_similarity -
            (1.0 - float(alpha)) * max_other)

    def _class_prototypes(self, exemplar_z, topk, alpha,
                          extra_exemplar_z=None):
        exemplar_z = F.normalize(exemplar_z, dim=1)
        if extra_exemplar_z is not None:
            if extra_exemplar_z.dim() != 3:
                raise ValueError(
                    'Extra exemplar embeddings must have shape [C, R, D]')
            expected = (
                exemplar_z.size(0), extra_exemplar_z.size(1),
                exemplar_z.size(1))
            if tuple(extra_exemplar_z.shape) != expected:
                raise ValueError(
                    'Extra exemplar embeddings do not align with anchors')
            extra_exemplar_z = F.normalize(extra_exemplar_z, dim=2)

        filled = int(self.queue_filled.item())
        neighbor_count = min(max(int(topk), 0), filled)
        neighbor_indices = None
        neighbor_valid = None
        memory = None
        if neighbor_count > 0:
            memory = F.normalize(
                self.queue[:, :filled].detach().t(), dim=1)
            similarity = torch.matmul(exemplar_z, memory.t())

            if exemplar_z.size(0) > 1:
                other_similarity = similarity.unsqueeze(0).expand(
                    exemplar_z.size(0), -1, -1).clone()
                diagonal = torch.eye(
                    exemplar_z.size(0), dtype=torch.bool,
                    device=exemplar_z.device)
                other_similarity[diagonal] = -float('inf')
                max_other = other_similarity.max(dim=1)[0]
            else:
                max_other = torch.zeros_like(similarity)

            score = (
                float(alpha) * similarity -
                (1.0 - float(alpha)) * max_other)
            if self.ose_prototype_stage == 0:
                neighbor_indices = torch.topk(
                    score, k=neighbor_count, dim=1).indices
                neighbor_valid = torch.ones_like(
                    neighbor_indices, dtype=torch.bool)
            else:
                # P1+: each queue slot is owned by exactly one class before
                # per-class Top-K.  Missing candidates stay padded with -1.
                owner = score.argmax(dim=0)
                neighbor_indices = torch.full(
                    (exemplar_z.size(0), neighbor_count), -1,
                    dtype=torch.long, device=exemplar_z.device)
                neighbor_valid = torch.zeros_like(
                    neighbor_indices, dtype=torch.bool)
                for class_index in range(exemplar_z.size(0)):
                    candidates = torch.nonzero(
                        owner == class_index, as_tuple=False).flatten()
                    selected_count = min(
                        neighbor_count, int(candidates.numel()))
                    if selected_count == 0:
                        continue
                    selected_in_candidates = torch.topk(
                        score[class_index, candidates],
                        k=selected_count).indices
                    selected = candidates[selected_in_candidates]
                    neighbor_indices[
                        class_index, :selected_count] = selected
                    neighbor_valid[
                        class_index, :selected_count] = True

            neighbor_sample_indices = torch.full_like(
                neighbor_indices, -1)
            neighbor_sample_indices[neighbor_valid] = (
                self.queue_sample_indices[:filled][
                    neighbor_indices[neighbor_valid]])
            neighbor_sample_indices = neighbor_sample_indices.detach()
        else:
            neighbor_sample_indices = torch.empty(
                exemplar_z.size(0), 0, dtype=torch.long,
                device=exemplar_z.device)
            neighbor_valid = torch.empty(
                exemplar_z.size(0), 0, dtype=torch.bool,
                device=exemplar_z.device)

        prototypes = []
        component_counts = []
        for class_index in range(exemplar_z.size(0)):
            component_groups = [
                exemplar_z[class_index:class_index + 1]]
            if extra_exemplar_z is not None:
                component_groups.append(extra_exemplar_z[class_index])
            if neighbor_count > 0:
                selected = neighbor_indices[class_index][
                    neighbor_valid[class_index]]
                if selected.numel() > 0:
                    component_groups.append(memory[selected])
            components = torch.cat(component_groups, dim=0)
            if self.ose_prototype_stage >= 2:
                aggregation_scores = self._component_competition_scores(
                    exemplar_z, components, class_index, alpha)
            else:
                aggregation_scores = torch.matmul(
                    components, exemplar_z[class_index])
            weights = torch.softmax(aggregation_scores, dim=0)
            prototype = torch.sum(
                weights.unsqueeze(1) * components, dim=0)
            if self.ose_prototype_stage >= 3:
                prototype = F.normalize(prototype, dim=0)
            prototypes.append(prototype)
            component_counts.append(components.size(0))

        valid_queue_indices = (
            neighbor_indices[neighbor_valid]
            if neighbor_indices is not None
            else torch.empty(
                0, dtype=torch.long, device=exemplar_z.device))
        if valid_queue_indices.numel() > 0:
            unique_count = torch.unique(valid_queue_indices).numel()
            overlap_rate = valid_queue_indices.new_tensor(
                1.0 - float(unique_count) /
                float(valid_queue_indices.numel()), dtype=exemplar_z.dtype)
        else:
            overlap_rate = exemplar_z.new_tensor(0.0)
        component_counts = torch.tensor(
            component_counts, dtype=torch.long, device=exemplar_z.device)
        return (torch.stack(prototypes, dim=0), neighbor_sample_indices,
                component_counts, overlap_rate)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, sample_indices=None):
        keys = keys.detach()
        if sample_indices is None:
            sample_indices = torch.full(
                (keys.size(0),), -1, dtype=torch.long, device=keys.device)
        else:
            sample_indices = sample_indices.detach().to(
                device=keys.device, dtype=torch.long).view(-1)
        if sample_indices.size(0) != keys.size(0):
            raise ValueError('Queue keys and sample indices must align')
        if keys.size(0) >= self.queue_size:
            keys = keys[-self.queue_size:]
            sample_indices = sample_indices[-self.queue_size:]

        count = keys.size(0)
        ptr = int(self.queue_ptr.item())
        first_count = min(count, self.queue_size - ptr)
        self.queue[:, ptr:ptr + first_count] = keys[:first_count].t()
        self.queue_sample_indices[ptr:ptr + first_count] = (
            sample_indices[:first_count])
        remaining = count - first_count
        if remaining > 0:
            self.queue[:, :remaining] = keys[first_count:].t()
            self.queue_sample_indices[:remaining] = (
                sample_indices[first_count:])

        self.queue_ptr[0] = (ptr + count) % self.queue_size
        self.queue_filled[0] = min(
            self.queue_size, int(self.queue_filled.item()) + count)

    @staticmethod
    def _category_confidence(category_target):
        category_target = category_target.detach()
        num_classes = category_target.size(1)
        if num_classes <= 1:
            return torch.ones(
                category_target.size(0), device=category_target.device,
                dtype=category_target.dtype)
        entropy = -(
            category_target * category_target.clamp_min(1e-12).log()
        ).sum(dim=1)
        return (1.0 - entropy / math.log(num_classes)).clamp(0.0, 1.0)

    @torch.no_grad()
    def _negative_weights(self, category_target):
        category_target = category_target.detach()
        if category_target.dim() != 2:
            raise ValueError('Category target must have shape [N, C]')
        if category_target.size(1) != self.category_queue.size(0):
            raise ValueError(
                'Category target and category queue class counts must match')

        current_confidence = self._category_confidence(category_target)
        queued_category = self.category_queue.clone().detach()
        queued_confidence = self.confidence_queue.clone().detach()
        category_similarity = torch.matmul(
            category_target, queued_category)
        negative_weight = 1.0 - (
            current_confidence.unsqueeze(1) *
            queued_confidence.unsqueeze(0) * category_similarity)
        return negative_weight.clamp(0.0, 1.0), current_confidence

    def _queue_contrastive_logits(self, query, positive_key,
                                  category_target):
        if not self.queue_contrast_enabled:
            raise ValueError('Queue contrast is disabled')
        query = F.normalize(query, dim=1)
        positive_key = F.normalize(positive_key.detach(), dim=1)
        positive_logits = torch.sum(
            query * positive_key, dim=1, keepdim=True)
        negative_similarity = torch.matmul(
            query, self.instance_queue.clone().detach())
        negative_weight, current_confidence = self._negative_weights(
            category_target)

        positive_logits = positive_logits / self.instance_temperature
        negative_logits = negative_similarity / self.instance_temperature
        negative_logits = negative_logits + torch.log(
            negative_weight.clamp_min(1e-6))
        logits = torch.cat([positive_logits, negative_logits], dim=1)
        return logits, negative_weight, current_confidence

    def _queue_contrastive_loss(self, online_features, teacher_features,
                                category_target):
        query = self.instance_projector_q(online_features)
        with torch.no_grad():
            positive_key = self.instance_projector_k(teacher_features)
        logits, negative_weight, current_confidence = (
            self._queue_contrastive_logits(
                query, positive_key, category_target.detach()))
        labels = torch.zeros(
            logits.size(0), dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(logits, labels)
        return (loss, positive_key.detach(), negative_weight.detach(),
                current_confidence.detach())

    @torch.no_grad()
    def _dequeue_and_enqueue_instance(self, keys, categories, confidence):
        if not self.queue_contrast_enabled:
            raise ValueError('Queue contrast is disabled')
        keys = F.normalize(keys.detach(), dim=1)
        categories = categories.detach()
        confidence = confidence.detach().view(-1)
        count = keys.size(0)
        if (categories.dim() != 2 or categories.size(0) != count or
                confidence.size(0) != count):
            raise ValueError(
                'Instance keys, categories, and confidence must align')
        if categories.size(1) != self.category_queue.size(0):
            raise ValueError(
                'Enqueued categories and category queue class counts must match')

        ptr = int(self.instance_queue_ptr.item())
        if count >= self.instance_queue_size:
            final_ptr = (ptr + count) % self.instance_queue_size
            keys = keys[-self.instance_queue_size:]
            categories = categories[-self.instance_queue_size:]
            confidence = confidence[-self.instance_queue_size:]
            count = self.instance_queue_size
            ptr = final_ptr

        first_count = min(count, self.instance_queue_size - ptr)
        self.instance_queue[:, ptr:ptr + first_count] = (
            keys[:first_count].t())
        self.category_queue[:, ptr:ptr + first_count] = (
            categories[:first_count].t())
        self.confidence_queue[ptr:ptr + first_count] = (
            confidence[:first_count])
        remaining = count - first_count
        if remaining > 0:
            self.instance_queue[:, :remaining] = keys[first_count:].t()
            self.category_queue[:, :remaining] = categories[first_count:].t()
            self.confidence_queue[:remaining] = confidence[first_count:]

        self.instance_queue_ptr[0] = (
            ptr + count) % self.instance_queue_size

    def _online_embeddings(self, view_a, view_b):
        view_a_features = self.encoder_q.forward_features(view_a)
        view_b_features = self.encoder_q.forward_features(view_b)

        view_a_projected = self.projector_q(view_a_features)
        view_b_projected = self.projector_q(view_b_features)
        view_a_prediction = F.normalize(
            self.predictor(view_a_projected), dim=1)
        view_b_prediction = F.normalize(
            self.predictor(view_b_projected), dim=1)

        features = [F.normalize(view_a_features, dim=1),
                    F.normalize(view_b_features, dim=1)]
        projections = [F.normalize(view_a_projected, dim=1),
                       F.normalize(view_b_projected, dim=1)]
        predictions = [view_a_prediction, view_b_prediction]
        return ([view_a_features, view_b_features], features,
                projections, predictions)

    def _exemplar_embedding(self, exemplar):
        return self._online_projection(exemplar)

    def _online_projection(self, view):
        features = self.encoder_q.forward_features(view)
        return F.normalize(self.projector_q(features), dim=1)

    @torch.no_grad()
    def _teacher_exemplar_projection(self, exemplar):
        modules = (self.encoder_k, self.projector_k)
        batch_norm_state = []
        for module in modules:
            for child in module.modules():
                if isinstance(child, (
                        nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    batch_norm_state.append((
                        child,
                        (child.running_mean.clone()
                         if child.running_mean is not None else None),
                        (child.running_var.clone()
                         if child.running_var is not None else None),
                        (child.num_batches_tracked.clone()
                         if child.num_batches_tracked is not None else None),
                    ))
        try:
            features = self.encoder_k.forward_features(exemplar)
            projected = self.projector_k(features)
        finally:
            for child, running_mean, running_var, batches in batch_norm_state:
                if running_mean is not None:
                    child.running_mean.copy_(running_mean)
                if running_var is not None:
                    child.running_var.copy_(running_var)
                if batches is not None:
                    child.num_batches_tracked.copy_(batches)
        return F.normalize(projected, dim=1)

    @torch.no_grad()
    def _teacher_embeddings(self, view_a, view_b):
        view_a_features = self.encoder_k.forward_features(view_a)
        view_b_features = self.encoder_k.forward_features(view_b)
        features = [F.normalize(view_a_features, dim=1),
                    F.normalize(view_b_features, dim=1)]
        embeddings = [
            F.normalize(self.projector_k(view_a_features), dim=1),
            F.normalize(self.projector_k(view_b_features), dim=1),
        ]
        return ([view_a_features, view_b_features], features, embeddings)

    def forward(self, view_a, view_b=None, exemplar=None,
                momentum=0.996, ose_topk=8, ose_alpha=0.75,
                ose_tau_s=0.1, ose_tau_t=0.04,
                sample_indices=None, mixed_view=None, mix_index=None,
                mix_beta=None, compute_mix_proto=False,
                compute_mix_ins=False, extra_exemplar_views=None):
        if not self.pretrain:
            return self.encoder_q(view_a)
        if view_b is None:
            raise ValueError('ReSA requires two view inputs')
        compute_mix_proto = bool(compute_mix_proto)
        compute_mix_ins = bool(compute_mix_ins)
        compute_mix = compute_mix_proto or compute_mix_ins
        if compute_mix and not self.ose_enabled:
            raise ValueError('Lmix requires OSE to be enabled')
        if compute_mix:
            if mixed_view is None or mix_index is None or mix_beta is None:
                raise ValueError(
                    'Lmix requires mixed_view, mix_index, and mix_beta')
        elif any(value is not None for value in (
                mixed_view, mix_index, mix_beta)):
            raise ValueError(
                'Mixed inputs were provided while both Lmix terms are disabled')

        online_raw_h, online_h, online_z, online_q = (
            self._online_embeddings(view_a, view_b))
        if self.ose_enabled:
            if exemplar is None:
                raise ValueError('ReSA+Lproto requires exemplar inputs')
            exemplar_z = self._exemplar_embedding(exemplar)
            if extra_exemplar_views is None:
                extra_exemplar_views = []
            if not isinstance(extra_exemplar_views, (tuple, list)):
                raise ValueError(
                    'extra_exemplar_views must be a list or tuple')
            for extra_view in extra_exemplar_views:
                if extra_view.size(0) != exemplar.size(0):
                    raise ValueError(
                        'All exemplar views must contain the same classes')
        with torch.no_grad():
            self._momentum_update(momentum)
            teacher_raw_h, teacher_h, teacher_z = self._teacher_embeddings(
                view_a, view_b)
            if self.ose_enabled and extra_exemplar_views:
                extra_exemplar_z = torch.stack([
                    self._teacher_exemplar_projection(extra_view)
                    for extra_view in extra_exemplar_views
                ], dim=1)
            else:
                extra_exemplar_z = None

        assignment = self._sinkhorn_knopp(
            torch.matmul(online_h[0].detach(), teacher_h[0].t()))
        cluster_loss = online_q[0].new_tensor(0.0)
        terms = 0
        for online_index in range(len(online_q)):
            for teacher_index in range(len(teacher_z)):
                if online_index == teacher_index:
                    continue
                logits = torch.matmul(
                    online_q[online_index], teacher_z[teacher_index].t())
                logits = logits / max(self.cluster_temperature, 1e-12)
                cluster_loss = cluster_loss + self._soft_cross_entropy(
                    logits, assignment)
                terms += 1
        cluster_loss = cluster_loss / max(terms, 1)
        cluster_entropy = -(
            assignment * assignment.clamp_min(1e-12).log()
        ).sum(dim=1).mean()

        result = {
            'cluster': cluster_loss,
            'cluster_entropy': cluster_entropy,
            'cluster_kl': cluster_loss - cluster_entropy,
        }
        if not self.ose_enabled:
            return result

        (prototypes, neighbor_sample_indices, prototype_component_counts,
         neighbor_overlap_rate) = self._class_prototypes(
             exemplar_z, topk=ose_topk, alpha=ose_alpha,
             extra_exemplar_z=extra_exemplar_z)
        student_logits = torch.matmul(
            online_z[1], prototypes.t()) / max(float(ose_tau_s), 1e-12)
        teacher_logits = torch.matmul(
            teacher_z[0].detach(), prototypes.detach().t()) / max(
                float(ose_tau_t), 1e-12)
        teacher_target = torch.softmax(teacher_logits, dim=1).detach()
        align_loss = self._soft_cross_entropy(student_logits, teacher_target)

        prototype_similarity = torch.matmul(prototypes, prototypes.t())
        off_diagonal = ~torch.eye(
            prototypes.size(0), dtype=torch.bool, device=prototypes.device)
        if prototypes.size(0) > 1:
            disp_loss = prototype_similarity[off_diagonal].mean()
            disp_loss = disp_loss / max(float(ose_tau_s), 1e-12)
        else:
            disp_loss = prototype_similarity.new_tensor(0.0)
        proto_loss = align_loss + disp_loss

        queue_corr_loss = proto_loss.new_tensor(0.0)
        mean_category_confidence = proto_loss.new_tensor(0.0)
        mean_negative_weight = proto_loss.new_tensor(1.0)
        min_negative_weight = proto_loss.new_tensor(1.0)
        instance_key = None
        instance_category = None
        instance_confidence = None
        if self.queue_contrast_enabled:
            # The positive key is EMA view_b.  Reuse the detached OSE teacher
            # category target for the same sample; labels never enter this path.
            instance_category = teacher_target.detach()
            (queue_corr_loss, instance_key, negative_weight,
             instance_confidence) = self._queue_contrastive_loss(
                online_raw_h[0], teacher_raw_h[1], instance_category)
            mean_category_confidence = instance_confidence.mean()
            mean_negative_weight = negative_weight.mean()
            min_negative_weight = negative_weight.min()

        mix_proto_loss = proto_loss.new_tensor(0.0)
        mix_ins_loss = proto_loss.new_tensor(0.0)
        if compute_mix:
            if mixed_view.size(0) != online_z[1].size(0):
                raise ValueError(
                    'Mixed view batch size must match the unlabeled views')
            mix_index = mix_index.detach().to(
                device=online_z[1].device, dtype=torch.long).view(-1)
            if mix_index.numel() != online_z[1].size(0):
                raise ValueError(
                    'Mix permutation size must match the batch size')
            if (mix_index.min().item() < 0 or
                    mix_index.max().item() >= online_z[1].size(0)):
                raise ValueError('Mix permutation contains invalid indices')
            mix_beta = float(mix_beta)
            if not 0.0 <= mix_beta <= 1.0:
                raise ValueError('Mix coefficient must be in [0, 1]')

            # The mixed branch remains in encoder-projector space. It does not
            # use the ReSA predictor, participate in Sinkhorn, or enter the queue.
            mixed_z = self._online_projection(mixed_view)
            if compute_mix_proto:
                mixed_logits = torch.matmul(
                    mixed_z, prototypes.t()) / max(
                        float(ose_tau_s), 1e-12)
                student_target = torch.softmax(
                    student_logits, dim=1).detach()
                mixed_target = (
                    mix_beta * student_target +
                    (1.0 - mix_beta) * teacher_target[mix_index])
                mix_proto_loss = self._soft_cross_entropy(
                    mixed_logits, mixed_target)

            if compute_mix_ins:
                instance_logits = torch.matmul(
                    mixed_z, teacher_z[0].detach().t()) / max(
                        float(ose_tau_s), 1e-12)
                instance_log_prob = F.log_softmax(instance_logits, dim=1)
                row = torch.arange(
                    mixed_z.size(0), device=mixed_z.device)
                mix_ins_loss = -(
                    mix_beta * instance_log_prob[row, row] +
                    (1.0 - mix_beta) *
                    instance_log_prob[row, mix_index]
                ).mean()

        target_entropy = -(
            teacher_target * teacher_target.clamp_min(1e-12).log()
        ).sum(dim=1).mean()
        self._dequeue_and_enqueue(teacher_z[0], sample_indices)
        if self.queue_contrast_enabled:
            self._dequeue_and_enqueue_instance(
                instance_key, instance_category, instance_confidence)

        result.update({
            'proto': proto_loss,
            'align': align_loss,
            'disp': disp_loss,
            'mix': mix_proto_loss + mix_ins_loss,
            'mix_proto': mix_proto_loss,
            'mix_ins': mix_ins_loss,
            'queue_corr': queue_corr_loss,
            'mean_category_confidence': mean_category_confidence,
            'mean_negative_weight': mean_negative_weight,
            'min_negative_weight': min_negative_weight,
            'target_entropy': target_entropy,
            'align_kl': align_loss - target_entropy,
            'queue_fill': self.queue_filled.float(),
            'instance_queue_ptr': (
                self.instance_queue_ptr.float()
                if self.queue_contrast_enabled
                else proto_loss.new_tensor(0.0)),
            'neighbor_sample_indices': neighbor_sample_indices,
            'prototype_component_counts': prototype_component_counts,
            'neighbor_overlap_rate': neighbor_overlap_rate,
        })
        return result
