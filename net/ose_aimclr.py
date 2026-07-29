import torch
import torch.nn.functional as F

from .aimclr import AimCLR


class OSEAimCLR(AimCLR):
    """AimCLR A2 with OSE prototypes and one shared feature queue.

    The native AimCLR embedding is used for instance contrast, exemplars,
    mutually exclusive P1 neighbors, prototypes, and interpolation losses.
    After mining starts, OSE first selects one semantic P1 pool and AimCLR's
    three query views choose at most one extra positive inside that pool.  No
    soft-positive weight, second projector, or second feature queue is used.
    """

    def __init__(self, base_encoder=None, pretrain=True, feature_dim=128,
                 queue_size=32768, momentum=0.999, Temperature=0.07,
                 mlp=True, in_channels=3, hidden_channels=64,
                 hidden_dim=256, num_class=60, dropout=0.5,
                 graph_args={'layout': 'ntu-rgb+d', 'strategy': 'spatial'},
                 edge_importance_weighting=True, ose_enabled=True,
                 # Accepted only so historical checkpoints/config objects can
                 # be inspected without leaking these obsolete values into the
                 # backbone constructor.
                 ose_feature_dim=None, ose_projector_hidden_dim=None,
                 ose_projector_layers=None, ose_queue_size=None, **kwargs):
        super().__init__(
            base_encoder=base_encoder, pretrain=pretrain,
            feature_dim=feature_dim, queue_size=queue_size,
            momentum=momentum, Temperature=Temperature, mlp=mlp,
            in_channels=in_channels, hidden_channels=hidden_channels,
            hidden_dim=hidden_dim, num_class=num_class, dropout=dropout,
            graph_args=graph_args,
            edge_importance_weighting=edge_importance_weighting, **kwargs)

        self.ose_enabled = bool(ose_enabled)
        if not self.pretrain:
            return

        # These are metadata sidecars for AimCLR's native queue, not another
        # feature memory.  OSE reads only slots that have received real EMA
        # keys; native AimCLR keeps its historical random-negative warm start.
        self.register_buffer(
            'queue_filled', torch.zeros(1, dtype=torch.long))
        self.register_buffer(
            'queue_sample_indices',
            torch.full((self.K,), -1, dtype=torch.long))

    @torch.no_grad()
    def reset_momentum_encoder(self):
        """Restore exact online/EMA initialization after global init."""
        if not self.pretrain:
            return
        for param_q, param_k in zip(
                self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

    @staticmethod
    def _soft_cross_entropy(logits, target):
        return -(target * F.log_softmax(logits, dim=1)).sum(dim=1).mean()

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, sample_indices=None):
        """Write EMA keys and OSE metadata into the one shared queue."""
        keys = keys.detach()
        if sample_indices is None:
            sample_indices = torch.full(
                (keys.size(0),), -1, dtype=torch.long, device=keys.device)
        else:
            sample_indices = sample_indices.detach().to(
                device=keys.device, dtype=torch.long).view(-1)
        if sample_indices.size(0) != keys.size(0):
            raise ValueError('Queue keys and sample indices must align')

        if keys.size(0) >= self.K:
            keys = keys[-self.K:]
            sample_indices = sample_indices[-self.K:]

        count = keys.size(0)
        ptr = int(self.queue_ptr.item())
        first_count = min(count, self.K - ptr)
        self.queue[:, ptr:ptr + first_count] = keys[:first_count].t()
        self.queue_sample_indices[ptr:ptr + first_count] = (
            sample_indices[:first_count])
        remaining = count - first_count
        if remaining > 0:
            self.queue[:, :remaining] = keys[first_count:].t()
            self.queue_sample_indices[:remaining] = (
                sample_indices[first_count:])
        self.queue_filled[0] = min(
            self.K, int(self.queue_filled.item()) + count)

    def _online_native_projection(self, view):
        features = self.encoder_q.forward_features(view)
        return F.normalize(self.encoder_q.fc(features), dim=1)

    @torch.no_grad()
    def _teacher_exemplar_projection(self, exemplar):
        """Encode an EMA exemplar view without changing EMA BN buffers."""
        batch_norm_state = []
        for child in self.encoder_k.modules():
            if isinstance(child, (
                    torch.nn.BatchNorm1d, torch.nn.BatchNorm2d,
                    torch.nn.BatchNorm3d)):
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
            projected = self.encoder_k.fc(features)
        finally:
            for child, running_mean, running_var, batches in batch_norm_state:
                if running_mean is not None:
                    child.running_mean.copy_(running_mean)
                if running_var is not None:
                    child.running_var.copy_(running_var)
                if batches is not None:
                    child.num_batches_tracked.copy_(batches)
        return F.normalize(projected, dim=1)

    def _class_prototypes(self, exemplar_z, topk, alpha,
                          extra_exemplar_z=None):
        """Build P1 prototypes using mutually exclusive shared-queue slots."""
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
        memory = None
        if neighbor_count > 0:
            memory = F.normalize(
                self.queue[:, :filled].detach().clone().t(), dim=1)
            similarity = torch.matmul(exemplar_z, memory.t())
            if exemplar_z.size(0) > 1:
                # The direct CxCxK expansion is nearly 0.5 GB for NTU60
                # and AimCLR's K=32768 queue.  Per-slot top-2 gives the same
                # max-over-other-classes value with only CxK storage.
                top_values, top_classes = torch.topk(
                    similarity, k=2, dim=0)
                class_indices = torch.arange(
                    exemplar_z.size(0), device=exemplar_z.device
                ).unsqueeze(1)
                max_other = torch.where(
                    top_classes[0].unsqueeze(0) == class_indices,
                    top_values[1].unsqueeze(0),
                    top_values[0].unsqueeze(0))
            else:
                max_other = torch.zeros_like(similarity)
            score = (
                float(alpha) * similarity -
                (1.0 - float(alpha)) * max_other)

            # P1: every queue slot receives one owner before per-class Top-K.
            owner = score.argmax(dim=0)
            neighbor_queue_indices = torch.full(
                (exemplar_z.size(0), neighbor_count), -1,
                dtype=torch.long, device=exemplar_z.device)
            neighbor_valid = torch.zeros_like(
                neighbor_queue_indices, dtype=torch.bool)
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
                neighbor_queue_indices[
                    class_index, :selected_count] = selected
                neighbor_valid[class_index, :selected_count] = True

            neighbor_sample_indices = torch.full_like(
                neighbor_queue_indices, -1)
            neighbor_sample_indices[neighbor_valid] = (
                self.queue_sample_indices[:filled][
                    neighbor_queue_indices[neighbor_valid]])
        else:
            neighbor_queue_indices = torch.empty(
                exemplar_z.size(0), 0, dtype=torch.long,
                device=exemplar_z.device)
            neighbor_sample_indices = torch.empty_like(
                neighbor_queue_indices)
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
            selected = neighbor_queue_indices[class_index][
                neighbor_valid[class_index]]
            if selected.numel() > 0:
                component_groups.append(memory[selected])
            components = torch.cat(component_groups, dim=0)
            # Preserve the successful P1 aggregation: raw anchor similarity.
            aggregation_scores = torch.matmul(
                components, exemplar_z[class_index])
            weights = torch.softmax(aggregation_scores, dim=0)
            prototypes.append(torch.sum(
                weights.unsqueeze(1) * components, dim=0))
            component_counts.append(components.size(0))

        valid_slots = neighbor_queue_indices[neighbor_valid]
        if valid_slots.numel() > 0:
            overlap_rate = exemplar_z.new_tensor(
                1.0 - float(torch.unique(valid_slots).numel()) /
                float(valid_slots.numel()))
        else:
            overlap_rate = exemplar_z.new_tensor(0.0)

        return {
            'prototypes': torch.stack(prototypes, dim=0),
            'neighbor_queue_indices': neighbor_queue_indices.detach(),
            'neighbor_sample_indices': neighbor_sample_indices.detach(),
            'neighbor_valid': neighbor_valid.detach(),
            'component_counts': torch.tensor(
                component_counts, dtype=torch.long,
                device=exemplar_z.device),
            'overlap_rate': overlap_rate,
        }

    @torch.no_grad()
    def _ose_constrained_nnm_mask(
            self, teacher_target, prototype_state,
            l_neg, l_neg_e, l_neg_ed, sample_indices=None):
        """Choose at most one queue positive from the predicted P1 pool.

        OSE determines the semantic candidate pool with the EMA weak view.
        AimCLR then ranks that small pool by the maximum similarity across its
        normal, extreme, and dropped-extreme query streams.  The current
        sample's historical queue entries are excluded when indices exist.
        """
        batch_size = teacher_target.size(0)
        positive_mask = torch.zeros_like(l_neg)
        predicted_classes = teacher_target.argmax(dim=1)
        selected_queue_indices = torch.full(
            (batch_size,), -1, dtype=torch.long,
            device=teacher_target.device)
        candidate_counts = torch.zeros(
            batch_size, dtype=torch.long, device=teacher_target.device)
        same_sample_filtered = torch.zeros(
            (), dtype=torch.long, device=teacher_target.device)

        if sample_indices is not None:
            sample_indices = sample_indices.detach().to(
                device=teacher_target.device, dtype=torch.long).view(-1)
            if sample_indices.numel() != batch_size:
                raise ValueError(
                    'Sample indices must match the contrastive batch size')

        neighbor_indices = prototype_state['neighbor_queue_indices']
        neighbor_valid = prototype_state['neighbor_valid']
        multi_view_similarity = torch.maximum(
            l_neg, torch.maximum(l_neg_e, l_neg_ed))

        for class_index in range(teacher_target.size(1)):
            candidates = neighbor_indices[class_index][
                neighbor_valid[class_index]]
            rows = torch.nonzero(
                predicted_classes == class_index,
                as_tuple=False).flatten()
            if candidates.numel() == 0 or rows.numel() == 0:
                continue
            candidate_scores = multi_view_similarity[
                rows.unsqueeze(1), candidates.unsqueeze(0)]
            candidate_valid = torch.ones_like(
                candidate_scores, dtype=torch.bool)
            if sample_indices is not None:
                current_indices = sample_indices[rows].unsqueeze(1)
                same_sample = (
                    current_indices >= 0) & (
                    self.queue_sample_indices[candidates].unsqueeze(0) ==
                    current_indices)
                candidate_valid &= ~same_sample
                same_sample_filtered += same_sample.sum()

            row_candidate_counts = candidate_valid.sum(dim=1)
            candidate_counts[rows] = row_candidate_counts
            rows_with_candidate = row_candidate_counts > 0
            valid_scores = candidate_scores.masked_fill(
                ~candidate_valid, float('-inf'))
            best_in_pool = valid_scores.argmax(dim=1)
            selected = candidates[best_in_pool[rows_with_candidate]]
            selected_rows = rows[rows_with_candidate]
            positive_mask[selected_rows, selected] = 1.0
            selected_queue_indices[selected_rows] = selected

        selected_sample_indices = torch.full_like(
            selected_queue_indices, -1)
        selected_valid = selected_queue_indices >= 0
        selected_sample_indices[selected_valid] = (
            self.queue_sample_indices[
                selected_queue_indices[selected_valid]])
        return positive_mask.detach(), {
            'nnm_predicted_classes': predicted_classes.detach(),
            'nnm_selected_queue_indices': selected_queue_indices.detach(),
            'nnm_selected_sample_indices': selected_sample_indices.detach(),
            'nnm_candidate_count': candidate_counts.float().mean(),
            'nnm_positive_rate': selected_valid.float().mean(),
            'nnm_same_sample_filtered': same_sample_filtered.float(),
        }

    def _ose_losses(self, q, k, exemplar_z, extra_exemplar_z,
                    mixed_view, mix_index, mix_beta, compute_mix_proto,
                    compute_mix_ins, topk, alpha, tau_s, tau_t):
        prototype_state = self._class_prototypes(
            exemplar_z, topk=topk, alpha=alpha,
            extra_exemplar_z=extra_exemplar_z)
        prototypes = prototype_state['prototypes']

        student_logits = torch.matmul(
            q, prototypes.t()) / max(float(tau_s), 1e-12)
        teacher_logits = torch.matmul(
            k.detach(), prototypes.detach().t()) / max(
                float(tau_t), 1e-12)
        teacher_target = torch.softmax(teacher_logits, dim=1).detach()
        align_loss = self._soft_cross_entropy(
            student_logits, teacher_target)

        prototype_similarity = torch.matmul(prototypes, prototypes.t())
        off_diagonal = ~torch.eye(
            prototypes.size(0), dtype=torch.bool,
            device=prototypes.device)
        if prototypes.size(0) > 1:
            disp_loss = prototype_similarity[off_diagonal].mean()
            disp_loss = disp_loss / max(float(tau_s), 1e-12)
        else:
            disp_loss = prototype_similarity.new_tensor(0.0)
        proto_loss = align_loss + disp_loss

        mix_proto_loss = proto_loss.new_tensor(0.0)
        mix_ins_loss = proto_loss.new_tensor(0.0)
        if compute_mix_proto or compute_mix_ins:
            if mixed_view.size(0) != q.size(0):
                raise ValueError(
                    'Mixed view batch size must match weak views')
            mix_index = mix_index.detach().to(
                device=q.device, dtype=torch.long).view(-1)
            if mix_index.numel() != q.size(0):
                raise ValueError(
                    'Mix permutation size must match batch size')
            if (mix_index.min().item() < 0 or
                    mix_index.max().item() >= q.size(0)):
                raise ValueError('Mix permutation contains invalid indices')
            mix_beta = float(mix_beta)
            if not 0.0 <= mix_beta <= 1.0:
                raise ValueError('Mix coefficient must be in [0, 1]')

            mixed_z = self._online_native_projection(mixed_view)
            if compute_mix_proto:
                mixed_logits = torch.matmul(
                    mixed_z, prototypes.t()) / max(
                        float(tau_s), 1e-12)
                student_target = torch.softmax(
                    student_logits, dim=1).detach()
                mixed_target = (
                    mix_beta * student_target +
                    (1.0 - mix_beta) * teacher_target[mix_index])
                mix_proto_loss = self._soft_cross_entropy(
                    mixed_logits, mixed_target)
            if compute_mix_ins:
                instance_logits = torch.matmul(
                    mixed_z, k.detach().t()) / max(
                        float(tau_s), 1e-12)
                instance_log_prob = F.log_softmax(
                    instance_logits, dim=1)
                row = torch.arange(mixed_z.size(0), device=mixed_z.device)
                mix_ins_loss = -(
                    mix_beta * instance_log_prob[row, row] +
                    (1.0 - mix_beta) *
                    instance_log_prob[row, mix_index]
                ).mean()

        target_entropy = -(
            teacher_target * teacher_target.clamp_min(1e-12).log()
        ).sum(dim=1).mean()
        result = {
            'proto': proto_loss,
            'align': align_loss,
            'disp': disp_loss,
            'mix': mix_proto_loss + mix_ins_loss,
            'mix_proto': mix_proto_loss,
            'mix_ins': mix_ins_loss,
            'target_entropy': target_entropy,
            'align_kl': align_loss - target_entropy,
            'queue_fill': self.queue_filled.float(),
            'neighbor_queue_indices': (
                prototype_state['neighbor_queue_indices']),
            'neighbor_sample_indices': (
                prototype_state['neighbor_sample_indices']),
            'prototype_component_counts': (
                prototype_state['component_counts']),
            'neighbor_overlap_rate': prototype_state['overlap_rate'],
        }
        return result, teacher_target, prototype_state

    def forward(self, im_q_extreme, im_q, im_k=None, nnm=False, topk=1,
                exemplar=None, mixed_view=None, mix_index=None, mix_beta=None,
                compute_ose=False, compute_mix_proto=False,
                compute_mix_ins=False, extra_exemplar_views=None,
                sample_indices=None, ose_topk=4, ose_alpha=0.75,
                ose_tau_s=0.1, ose_tau_t=0.04):
        if not self.pretrain:
            return self.encoder_q(im_q)
        if im_q is None or im_k is None or im_q_extreme is None:
            raise ValueError('AimCLR pretraining requires three input views')

        compute_ose = bool(compute_ose)
        compute_mix_proto = bool(compute_mix_proto)
        compute_mix_ins = bool(compute_mix_ins)
        compute_mix = compute_mix_proto or compute_mix_ins
        if compute_ose and not self.ose_enabled:
            raise ValueError('OSE losses cannot run when OSE is disabled')
        if compute_mix and not compute_ose:
            raise ValueError('Lmix requires active OSE')
        if compute_mix:
            if mixed_view is None or mix_index is None or mix_beta is None:
                raise ValueError(
                    'Lmix requires mixed_view, mix_index, and mix_beta')
        elif any(value is not None for value in (
                mixed_view, mix_index, mix_beta)):
            raise ValueError(
                'Mixed inputs require at least one enabled Lmix term')
        q_features = self.encoder_q.forward_features(im_q)
        q = F.normalize(self.encoder_q.fc(q_features), dim=1)
        q_extreme_features, q_extreme_drop_features = (
            self.encoder_q.forward_features(im_q_extreme, drop=True))
        q_extreme = F.normalize(
            self.encoder_q.fc(q_extreme_features), dim=1)
        q_extreme_drop = F.normalize(
            self.encoder_q.fc(q_extreme_drop_features), dim=1)

        if compute_ose:
            if exemplar is None:
                raise ValueError('AimCLR A2 requires exemplar inputs')
            exemplar_z = self._online_native_projection(exemplar)
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
            self._momentum_update_key_encoder()
            k_features = self.encoder_k.forward_features(im_k)
            k = F.normalize(self.encoder_k.fc(k_features), dim=1)
            if compute_ose and extra_exemplar_views:
                extra_exemplar_z = torch.stack([
                    self._teacher_exemplar_projection(extra_view)
                    for extra_view in extra_exemplar_views
                ], dim=1)
            else:
                extra_exemplar_z = None

        queue_snapshot = self.queue.clone().detach()
        l_pos = torch.sum(q * k, dim=1, keepdim=True)
        l_neg = torch.matmul(q, queue_snapshot)
        l_pos_e = torch.sum(q_extreme * k, dim=1, keepdim=True)
        l_neg_e = torch.matmul(q_extreme, queue_snapshot)
        l_pos_ed = torch.sum(q_extreme_drop * k, dim=1, keepdim=True)
        l_neg_ed = torch.matmul(q_extreme_drop, queue_snapshot)

        logits = torch.cat([l_pos, l_neg], dim=1) / self.T
        logits_e = torch.cat([l_pos_e, l_neg_e], dim=1) / self.T
        logits_ed = torch.cat([l_pos_ed, l_neg_ed], dim=1) / self.T
        logits_e_prob = torch.softmax(logits_e, dim=1)
        logits_ed_prob = torch.softmax(logits_ed, dim=1)
        labels_ddm = torch.softmax(logits.detach(), dim=1)

        queue_positive_mask = torch.zeros_like(l_neg)
        if nnm and not compute_ose:
            _, topkdix = torch.topk(l_neg, topk, dim=1)
            _, topkdix_e = torch.topk(l_neg_e, topk, dim=1)
            _, topkdix_ed = torch.topk(l_neg_ed, topk, dim=1)
            queue_positive_mask.scatter_(1, topkdix, 1.0)
            queue_positive_mask.scatter_(1, topkdix_e, 1.0)
            queue_positive_mask.scatter_(1, topkdix_ed, 1.0)

        ose_losses = None
        if compute_ose:
            ose_losses, teacher_target, prototype_state = self._ose_losses(
                q=q, k=k, exemplar_z=exemplar_z,
                extra_exemplar_z=extra_exemplar_z,
                mixed_view=mixed_view, mix_index=mix_index,
                mix_beta=mix_beta,
                compute_mix_proto=compute_mix_proto,
                compute_mix_ins=compute_mix_ins, topk=ose_topk,
                alpha=ose_alpha, tau_s=ose_tau_s, tau_t=ose_tau_t)
            if nnm:
                queue_positive_mask, nnm_state = (
                    self._ose_constrained_nnm_mask(
                        teacher_target=teacher_target,
                        prototype_state=prototype_state,
                        l_neg=l_neg, l_neg_e=l_neg_e,
                        l_neg_ed=l_neg_ed,
                        sample_indices=sample_indices))
                ose_losses.update(nnm_state)
            else:
                ose_losses.update({
                    'nnm_predicted_classes': teacher_target.argmax(
                        dim=1).detach(),
                    'nnm_selected_queue_indices': torch.full(
                        (q.size(0),), -1, dtype=torch.long,
                        device=q.device),
                    'nnm_selected_sample_indices': torch.full(
                        (q.size(0),), -1, dtype=torch.long,
                        device=q.device),
                    'nnm_candidate_count': q.new_tensor(0.0),
                    'nnm_positive_rate': q.new_tensor(0.0),
                    'nnm_same_sample_filtered': q.new_tensor(0.0),
                })

        if nnm:
            contrast_target = torch.cat([
                torch.ones(
                    q.size(0), 1, device=q.device, dtype=q.dtype),
                queue_positive_mask,
            ], dim=1)
        else:
            contrast_target = torch.zeros(
                q.size(0), dtype=torch.long, device=q.device)

        # Every target and loss above reads the old queue.  The current EMA
        # weak key is inserted exactly once only after those values exist.
        self._dequeue_and_enqueue(k, sample_indices)
        return (logits, contrast_target, logits_e_prob, logits_ed_prob,
                labels_ddm, ose_losses)
