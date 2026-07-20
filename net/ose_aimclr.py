import torch
import torch.nn as nn
import torch.nn.functional as F

from .aimclr import AimCLR
from .ose_resa import _build_projector


class OSEAimCLR(AimCLR):
    """AimCLR with an independent modern projection space for OSE.

    AimCLR retains its native instance head and negative queue.  OSE operates
    only on encoder features through its own online/EMA projectors and, when
    requested, its own neighbor queue.  This separation is important: the two
    objectives share the backbone, but no head or stored tensor has two roles.
    """

    def __init__(self, base_encoder=None, pretrain=True, feature_dim=128,
                 queue_size=32768, momentum=0.999, Temperature=0.07, mlp=True,
                 in_channels=3, hidden_channels=64, hidden_dim=256,
                 num_class=60, dropout=0.5,
                 graph_args={'layout': 'ntu-rgb+d', 'strategy': 'spatial'},
                 edge_importance_weighting=True, ose_enabled=True,
                 ose_feature_dim=256, ose_projector_hidden_dim=2048,
                 ose_projector_layers=3, ose_queue_size=8192, **kwargs):
        super().__init__(
            base_encoder=base_encoder, pretrain=pretrain,
            feature_dim=feature_dim, queue_size=queue_size,
            momentum=momentum, Temperature=Temperature, mlp=mlp,
            in_channels=in_channels, hidden_channels=hidden_channels,
            hidden_dim=hidden_dim, num_class=num_class, dropout=dropout,
            graph_args=graph_args,
            edge_importance_weighting=edge_importance_weighting, **kwargs)

        self.ose_enabled = bool(ose_enabled)
        if not self.pretrain or not self.ose_enabled:
            return

        self.ose_queue_size = int(ose_queue_size)
        if self.ose_queue_size <= 0:
            raise ValueError('ose_queue_size must be positive')

        self.ose_projector_q = _build_projector(
            hidden_dim, ose_projector_hidden_dim, ose_feature_dim,
            ose_projector_layers)
        self.ose_projector_k = _build_projector(
            hidden_dim, ose_projector_hidden_dim, ose_feature_dim,
            ose_projector_layers)

        self.register_buffer(
            'ose_queue', torch.zeros(ose_feature_dim, self.ose_queue_size))
        self.register_buffer(
            'ose_queue_ptr', torch.zeros(1, dtype=torch.long))
        self.register_buffer(
            'ose_queue_filled', torch.zeros(1, dtype=torch.long))
        self.register_buffer(
            'ose_queue_sample_indices',
            torch.full((self.ose_queue_size,), -1, dtype=torch.long))
        self.reset_ose_momentum_projector()

    @torch.no_grad()
    def reset_ose_momentum_projector(self):
        """Copy the online OSE projector after global weight initialization."""
        if not self.pretrain or not self.ose_enabled:
            return
        for param_q, param_k in zip(self.ose_projector_q.parameters(),
                                    self.ose_projector_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        super()._momentum_update_key_encoder()
        if not self.ose_enabled:
            return
        for param_q, param_k in zip(self.ose_projector_q.parameters(),
                                    self.ose_projector_k.parameters()):
            param_k.data.mul_(self.m).add_(
                param_q.data, alpha=1.0 - self.m)

    @staticmethod
    def _soft_cross_entropy(logits, target):
        return -(target * F.log_softmax(logits, dim=1)).sum(dim=1).mean()

    def _class_prototypes(self, exemplar_z, topk, alpha,
                          extra_exemplar_z=None):
        """Build prototypes from an online anchor, EMA views, and/or neighbors."""
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

        filled = int(self.ose_queue_filled.item())
        neighbor_count = min(max(int(topk), 0), filled)
        neighbor_indices = None
        memory = None
        if neighbor_count > 0:
            memory = F.normalize(
                self.ose_queue[:, :filled].detach().t(), dim=1)
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
            neighbor_indices = torch.topk(
                score, k=neighbor_count, dim=1).indices
            neighbor_sample_indices = self.ose_queue_sample_indices[
                :filled][neighbor_indices].detach()
        else:
            neighbor_sample_indices = torch.empty(
                exemplar_z.size(0), 0, dtype=torch.long,
                device=exemplar_z.device)

        extra_count = (extra_exemplar_z.size(1)
                       if extra_exemplar_z is not None else 0)
        component_count = 1 + extra_count + neighbor_count
        prototypes = []
        for class_index in range(exemplar_z.size(0)):
            component_groups = [
                exemplar_z[class_index:class_index + 1]]
            if extra_exemplar_z is not None:
                component_groups.append(extra_exemplar_z[class_index])
            if neighbor_count > 0:
                component_groups.append(
                    memory[neighbor_indices[class_index]])
            components = torch.cat(component_groups, dim=0)
            weights = torch.softmax(
                torch.matmul(components, exemplar_z[class_index]), dim=0)
            prototypes.append(
                torch.sum(weights.unsqueeze(1) * components, dim=0))
        return (torch.stack(prototypes, dim=0),
                neighbor_sample_indices, component_count)

    @torch.no_grad()
    def _dequeue_and_enqueue_ose(self, keys, sample_indices=None):
        keys = keys.detach()
        if sample_indices is None:
            sample_indices = torch.full(
                (keys.size(0),), -1, dtype=torch.long, device=keys.device)
        else:
            sample_indices = sample_indices.detach().to(
                device=keys.device, dtype=torch.long).view(-1)
        if sample_indices.size(0) != keys.size(0):
            raise ValueError('OSE queue keys and sample indices must align')
        if keys.size(0) >= self.ose_queue_size:
            keys = keys[-self.ose_queue_size:]
            sample_indices = sample_indices[-self.ose_queue_size:]

        count = keys.size(0)
        ptr = int(self.ose_queue_ptr.item())
        first_count = min(count, self.ose_queue_size - ptr)
        self.ose_queue[:, ptr:ptr + first_count] = keys[:first_count].t()
        self.ose_queue_sample_indices[ptr:ptr + first_count] = (
            sample_indices[:first_count])
        remaining = count - first_count
        if remaining > 0:
            self.ose_queue[:, :remaining] = keys[first_count:].t()
            self.ose_queue_sample_indices[:remaining] = (
                sample_indices[first_count:])

        self.ose_queue_ptr[0] = (
            ptr + count) % self.ose_queue_size
        self.ose_queue_filled[0] = min(
            self.ose_queue_size,
            int(self.ose_queue_filled.item()) + count)

    def _online_ose_projection(self, view):
        features = self.encoder_q.forward_features(view)
        return F.normalize(self.ose_projector_q(features), dim=1)

    @torch.no_grad()
    def _teacher_exemplar_projection(self, exemplar):
        """Project an EMA exemplar view without repeated BN-buffer updates."""
        modules = (self.encoder_k, self.ose_projector_k)
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
            projected = self.ose_projector_k(features)
        finally:
            for child, running_mean, running_var, batches in batch_norm_state:
                if running_mean is not None:
                    child.running_mean.copy_(running_mean)
                if running_var is not None:
                    child.running_var.copy_(running_var)
                if batches is not None:
                    child.num_batches_tracked.copy_(batches)
        return F.normalize(projected, dim=1)

    def _ose_losses(self, q_ose, k_ose, exemplar_z, extra_exemplar_z,
                    mixed_view, mix_index, mix_beta, compute_mix_proto,
                    compute_mix_ins, topk, alpha, tau_s, tau_t,
                    sample_indices):
        (prototypes, neighbor_sample_indices,
         prototype_component_count) = self._class_prototypes(
            exemplar_z, topk=topk, alpha=alpha,
            extra_exemplar_z=extra_exemplar_z)
        student_logits = torch.matmul(
            q_ose, prototypes.t()) / max(float(tau_s), 1e-12)
        teacher_logits = torch.matmul(
            k_ose.detach(), prototypes.detach().t()) / max(
                float(tau_t), 1e-12)
        teacher_target = torch.softmax(teacher_logits, dim=1).detach()
        align_loss = self._soft_cross_entropy(
            student_logits, teacher_target)

        prototype_similarity = torch.matmul(prototypes, prototypes.t())
        off_diagonal = ~torch.eye(
            prototypes.size(0), dtype=torch.bool, device=prototypes.device)
        if prototypes.size(0) > 1:
            disp_loss = prototype_similarity[off_diagonal].mean()
            disp_loss = disp_loss / max(float(tau_s), 1e-12)
        else:
            disp_loss = prototype_similarity.new_tensor(0.0)
        proto_loss = align_loss + disp_loss

        mix_proto_loss = proto_loss.new_tensor(0.0)
        mix_ins_loss = proto_loss.new_tensor(0.0)
        compute_mix = bool(compute_mix_proto) or bool(compute_mix_ins)
        if compute_mix:
            if mixed_view.size(0) != q_ose.size(0):
                raise ValueError(
                    'Mixed view batch size must match the unlabeled views')
            mix_index = mix_index.detach().to(
                device=q_ose.device, dtype=torch.long).view(-1)
            if mix_index.numel() != q_ose.size(0):
                raise ValueError(
                    'Mix permutation size must match the batch size')
            if (mix_index.min().item() < 0 or
                    mix_index.max().item() >= q_ose.size(0)):
                raise ValueError('Mix permutation contains invalid indices')
            mix_beta = float(mix_beta)
            if not 0.0 <= mix_beta <= 1.0:
                raise ValueError('Mix coefficient must be in [0, 1]')

            # This branch uses only encoder_q -> ose_projector_q.  It never
            # enters the AimCLR head, either queue, or any teacher path.
            mixed_z = self._online_ose_projection(mixed_view)
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
                    mixed_z, k_ose.detach().t()) / max(
                        float(tau_s), 1e-12)
                instance_log_prob = F.log_softmax(
                    instance_logits, dim=1)
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
        if int(topk) > 0:
            self._dequeue_and_enqueue_ose(k_ose, sample_indices)

        return {
            'proto': proto_loss,
            'align': align_loss,
            'disp': disp_loss,
            'mix': mix_proto_loss + mix_ins_loss,
            'mix_proto': mix_proto_loss,
            'mix_ins': mix_ins_loss,
            'target_entropy': target_entropy,
            'align_kl': align_loss - target_entropy,
            'queue_fill': self.ose_queue_filled.float(),
            'neighbor_sample_indices': neighbor_sample_indices,
            'prototype_components': proto_loss.new_tensor(
                prototype_component_count, dtype=torch.long),
        }

    def forward(self, im_q_extreme, im_q, im_k=None, nnm=False, topk=1,
                exemplar=None, mixed_view=None, mix_index=None, mix_beta=None,
                compute_ose=False, compute_mix_proto=False,
                compute_mix_ins=False, extra_exemplar_views=None,
                sample_indices=None, ose_topk=8, ose_alpha=0.75,
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
            raise ValueError('Lmix requires OSE losses to be active')
        if compute_mix:
            if mixed_view is None or mix_index is None or mix_beta is None:
                raise ValueError(
                    'Lmix requires mixed_view, mix_index, and mix_beta')
        elif any(value is not None for value in (
                mixed_view, mix_index, mix_beta)):
            raise ValueError(
                'Mixed inputs were provided while both Lmix terms are disabled')

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
                raise ValueError('AimCLR+Lproto requires exemplar inputs')
            q_ose = F.normalize(
                self.ose_projector_q(q_features), dim=1)
            exemplar_z = self._online_ose_projection(exemplar)
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
            if compute_ose:
                k_ose = F.normalize(
                    self.ose_projector_k(k_features), dim=1)
                if extra_exemplar_views:
                    extra_exemplar_z = torch.stack([
                        self._teacher_exemplar_projection(extra_view)
                        for extra_view in extra_exemplar_views
                    ], dim=1)
                else:
                    extra_exemplar_z = None

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum(
            'nc,ck->nk', [q, self.queue.clone().detach()])
        l_pos_e = torch.einsum(
            'nc,nc->n', [q_extreme, k]).unsqueeze(-1)
        l_neg_e = torch.einsum(
            'nc,ck->nk', [q_extreme, self.queue.clone().detach()])
        l_pos_ed = torch.einsum(
            'nc,nc->n', [q_extreme_drop, k]).unsqueeze(-1)
        l_neg_ed = torch.einsum(
            'nc,ck->nk', [
                q_extreme_drop, self.queue.clone().detach()])

        logits = torch.cat([l_pos, l_neg], dim=1) / self.T
        logits_e = torch.cat([l_pos_e, l_neg_e], dim=1) / self.T
        logits_ed = torch.cat([l_pos_ed, l_neg_ed], dim=1) / self.T
        logits_e_prob = torch.softmax(logits_e, dim=1)
        logits_ed_prob = torch.softmax(logits_ed, dim=1)
        labels_ddm = torch.softmax(logits.detach(), dim=1)

        if nnm:
            _, topkdix = torch.topk(l_neg, topk, dim=1)
            _, topkdix_e = torch.topk(l_neg_e, topk, dim=1)
            _, topkdix_ed = torch.topk(l_neg_ed, topk, dim=1)
            topk_onehot = torch.zeros_like(l_neg)
            topk_onehot.scatter_(1, topkdix, 1)
            topk_onehot.scatter_(1, topkdix_e, 1)
            topk_onehot.scatter_(1, topkdix_ed, 1)
            first_pos = torch.ones(
                topk_onehot.size(0), 1, device=topk_onehot.device)
            contrast_target = torch.cat(
                [first_pos, topk_onehot], dim=1)
        else:
            contrast_target = torch.zeros(
                logits.shape[0], dtype=torch.long, device=logits.device)

        ose_losses = None
        if compute_ose:
            ose_losses = self._ose_losses(
                q_ose=q_ose, k_ose=k_ose, exemplar_z=exemplar_z,
                extra_exemplar_z=extra_exemplar_z,
                mixed_view=mixed_view, mix_index=mix_index,
                mix_beta=mix_beta,
                compute_mix_proto=compute_mix_proto,
                compute_mix_ins=compute_mix_ins, topk=ose_topk,
                alpha=ose_alpha, tau_s=ose_tau_s, tau_t=ose_tau_t,
                sample_indices=sample_indices)

        self._dequeue_and_enqueue(k)
        return (logits, contrast_target, logits_e_prob, logits_ed_prob,
                labels_ddm, ose_losses)
