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


class OSEResA(nn.Module):
    """ReSA with the exemplar-guided prototype loss from OSESSL."""

    def __init__(self, base_encoder=None, pretrain=True, feature_dim=256,
                 projector_hidden_dim=2048, projector_layers=3,
                 use_predictor=True, ose_enabled=True, queue_size=8192,
                 cluster_temperature=0.4, sinkhorn_temperature=0.05,
                 sinkhorn_iterations=3, in_channels=3, hidden_channels=16,
                 hidden_dim=256, num_class=60, dropout=0.5,
                 graph_args={'layout': 'ntu-rgb+d', 'strategy': 'spatial'},
                 edge_importance_weighting=True, **kwargs):
        super().__init__()
        base_encoder = import_class(base_encoder)
        self.pretrain = pretrain
        self.ose_enabled = bool(ose_enabled)

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

    @torch.no_grad()
    def _momentum_update(self, momentum):
        momentum = float(momentum)
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data.mul_(momentum).add_(param_q.data, alpha=1.0 - momentum)
        for param_q, param_k in zip(self.projector_q.parameters(),
                                    self.projector_k.parameters()):
            param_k.data.mul_(momentum).add_(param_q.data, alpha=1.0 - momentum)

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

    def _class_prototypes(self, exemplar_z, topk, alpha):
        exemplar_z = F.normalize(exemplar_z, dim=1)
        filled = int(self.queue_filled.item())
        if filled == 0:
            neighbor_sample_indices = torch.empty(
                exemplar_z.size(0), 0, dtype=torch.long,
                device=exemplar_z.device)
            return exemplar_z, neighbor_sample_indices

        memory = F.normalize(self.queue[:, :filled].detach().t(), dim=1)
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

        score = float(alpha) * similarity - (1.0 - float(alpha)) * max_other
        neighbor_count = min(int(topk), memory.size(0))
        neighbor_indices = torch.topk(
            score, k=neighbor_count, dim=1).indices
        neighbor_sample_indices = self.queue_sample_indices[
            :filled][neighbor_indices].detach()

        prototypes = []
        for class_index in range(exemplar_z.size(0)):
            neighbors = memory[neighbor_indices[class_index]]
            components = torch.cat(
                [exemplar_z[class_index:class_index + 1], neighbors], dim=0)
            weights = torch.softmax(
                torch.matmul(components, exemplar_z[class_index]), dim=0)
            prototypes.append(
                torch.sum(weights.unsqueeze(1) * components, dim=0))
        return torch.stack(prototypes, dim=0), neighbor_sample_indices

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

    def _online_embeddings(self, view_a, view_b):
        view_a_features = self.encoder_q.forward_features(view_a)
        view_b_features = self.encoder_q.forward_features(view_b)

        view_a_embedding = F.normalize(
            self.predictor(self.projector_q(view_a_features)), dim=1)
        view_b_embedding = F.normalize(
            self.predictor(self.projector_q(view_b_features)), dim=1)

        features = [F.normalize(view_a_features, dim=1),
                    F.normalize(view_b_features, dim=1)]
        embeddings = [view_a_embedding, view_b_embedding]
        return features, embeddings

    def _exemplar_embedding(self, exemplar):
        exemplar_features = self.encoder_q.forward_features(exemplar)
        return F.normalize(
            self.predictor(self.projector_q(exemplar_features)), dim=1)

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
        return features, embeddings

    def forward(self, view_a, view_b=None, exemplar=None,
                momentum=0.996, ose_topk=8, ose_alpha=0.75,
                ose_tau_s=0.1, ose_tau_t=0.04,
                sample_indices=None):
        if not self.pretrain:
            return self.encoder_q(view_a)
        if view_b is None:
            raise ValueError('ReSA requires two view inputs')

        online_h, online_z = self._online_embeddings(view_a, view_b)
        if self.ose_enabled:
            if exemplar is None:
                raise ValueError('ReSA+Lproto requires exemplar inputs')
            exemplar_z = self._exemplar_embedding(exemplar)
        with torch.no_grad():
            self._momentum_update(momentum)
            teacher_h, teacher_z = self._teacher_embeddings(
                view_a, view_b)

        assignment = self._sinkhorn_knopp(
            torch.matmul(online_h[0].detach(), teacher_h[0].t()))
        cluster_loss = online_z[0].new_tensor(0.0)
        terms = 0
        for online_index in range(len(online_z)):
            for teacher_index in range(len(teacher_z)):
                if online_index == teacher_index:
                    continue
                logits = torch.matmul(
                    online_z[online_index], teacher_z[teacher_index].t())
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

        prototypes, neighbor_sample_indices = self._class_prototypes(
            exemplar_z, topk=ose_topk, alpha=ose_alpha)
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

        target_entropy = -(
            teacher_target * teacher_target.clamp_min(1e-12).log()
        ).sum(dim=1).mean()
        self._dequeue_and_enqueue(teacher_z[0], sample_indices)

        result.update({
            'proto': proto_loss,
            'align': align_loss,
            'disp': disp_loss,
            'target_entropy': target_entropy,
            'align_kl': align_loss - target_entropy,
            'queue_fill': self.queue_filled.float(),
            'neighbor_sample_indices': neighbor_sample_indices,
        })
        return result
