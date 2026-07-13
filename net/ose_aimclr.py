import torch
import torch.nn as nn
import torch.nn.functional as F

from .aimclr import AimCLR


class OSEAimCLR(AimCLR):
    """AimCLR with an independent online/EMA projection space for OSE."""

    def __init__(self, base_encoder=None, pretrain=True, feature_dim=128, queue_size=32768,
                 momentum=0.999, Temperature=0.07, mlp=True, in_channels=3,
                 hidden_channels=64, hidden_dim=256, num_class=60, dropout=0.5,
                 graph_args={'layout': 'ntu-rgb+d', 'strategy': 'spatial'},
                 edge_importance_weighting=True, **kwargs):
        super().__init__(
            base_encoder=base_encoder, pretrain=pretrain, feature_dim=feature_dim,
            queue_size=queue_size, momentum=momentum, Temperature=Temperature,
            mlp=mlp, in_channels=in_channels, hidden_channels=hidden_channels,
            hidden_dim=hidden_dim, num_class=num_class, dropout=dropout,
            graph_args=graph_args, edge_importance_weighting=edge_importance_weighting,
            **kwargs)

        if not self.pretrain:
            return

        self.ose_projector_q = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim))
        self.ose_projector_k = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim))
        for param_k in self.ose_projector_k.parameters():
            param_k.requires_grad = False

        self.register_buffer(
            'ose_queue', F.normalize(torch.randn(feature_dim, queue_size), dim=0))
        self.register_buffer('ose_queue_ptr', torch.zeros(1, dtype=torch.long))
        self.register_buffer('ose_initialized', torch.zeros(1, dtype=torch.uint8))

    @torch.no_grad()
    def activate_ose(self):
        """Initialize OSE from the trained AimCLR space at the warmup boundary."""
        if bool(self.ose_initialized.item()):
            return
        try:
            self.ose_projector_q.load_state_dict(self.encoder_q.fc.state_dict())
            self.ose_projector_k.load_state_dict(self.encoder_k.fc.state_dict())
        except RuntimeError as exc:
            raise RuntimeError(
                'OSE requires the two-layer AimCLR MLP projection head') from exc
        self.ose_queue.copy_(self.queue)
        self.ose_queue_ptr.copy_(self.queue_ptr)
        self.ose_initialized.fill_(1)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        super()._momentum_update_key_encoder()
        for param_q, param_k in zip(self.ose_projector_q.parameters(),
                                    self.ose_projector_k.parameters()):
            param_k.data.mul_(self.m).add_(param_q.data, alpha=1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue_ose(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.ose_queue_ptr)
        gpu_index = keys.device.index
        start = ptr if gpu_index is None else ptr + batch_size * gpu_index
        self.ose_queue[:, start:start + batch_size] = keys.T

    @torch.no_grad()
    def update_ptr(self, batch_size):
        super().update_ptr(batch_size)
        self.ose_queue_ptr[0] = (self.ose_queue_ptr[0] + batch_size) % self.K

    @staticmethod
    def _soft_cross_entropy(logits, target):
        return -torch.mean(torch.sum(F.log_softmax(logits, dim=1) * target, dim=1))

    def _class_prototypes(self, exemplar_z, topk=8, alpha=0.75, nn_idx=None):
        memory = F.normalize(self.ose_queue.clone().detach().t(), dim=1)
        exemplar_z = F.normalize(exemplar_z, dim=1)

        if nn_idx is None:
            sim = torch.matmul(exemplar_z, memory.t())
            if exemplar_z.size(0) > 1:
                other_sim = sim.unsqueeze(0).expand(
                    exemplar_z.size(0), -1, -1).clone()
                eye = torch.eye(
                    exemplar_z.size(0), dtype=torch.bool, device=sim.device)
                other_sim[eye] = -float('inf')
                max_other = other_sim.max(dim=1)[0]
            else:
                max_other = sim.new_zeros(sim.size())
            score = float(alpha) * sim - (1.0 - float(alpha)) * max_other
            topk = min(int(topk), memory.size(0))
            _, nn_idx = torch.topk(score, k=topk, dim=1)

        prototypes = []
        for class_idx in range(exemplar_z.size(0)):
            neighbors = memory[nn_idx[class_idx]]
            components = torch.cat(
                [exemplar_z[class_idx:class_idx + 1], neighbors], dim=0)
            weights = torch.softmax(
                torch.matmul(components, exemplar_z[class_idx]), dim=0)
            prototypes.append(
                torch.sum(weights.unsqueeze(1) * components, dim=0))
        return F.normalize(torch.stack(prototypes, dim=0), dim=1), nn_idx

    def _student_exemplar_features(self, exemplar):
        """Retain gradients but prevent exemplars from changing online BN statistics."""
        was_training = self.encoder_q.training
        self.encoder_q.eval()
        try:
            features = self.encoder_q.forward_features(exemplar)
        finally:
            self.encoder_q.train(was_training)
        return F.normalize(self.ose_projector_q(features), dim=1)

    @torch.no_grad()
    def _key_exemplar_features(self, exemplar):
        """Generate teacher exemplars without changing key-encoder BN statistics."""
        was_training = self.encoder_k.training
        self.encoder_k.eval()
        try:
            features = self.encoder_k.forward_features(exemplar)
        finally:
            self.encoder_k.train(was_training)
        return F.normalize(self.ose_projector_k(features), dim=1)

    def _ose_losses(self, q_ose, k_ose, exemplar, z_mix, mix_index, mix_beta,
                    topk=8, alpha=0.75, tau_s=0.04, tau_t=0.1):
        key_exemplar_z = self._key_exemplar_features(exemplar)
        key_prototypes, nn_idx = self._class_prototypes(
            key_exemplar_z, topk=topk, alpha=alpha)
        key_prototypes = key_prototypes.detach()

        p_q_logits = torch.matmul(q_ose, key_prototypes.t()) / max(
            float(tau_s), 1e-12)
        p_k_logits = torch.matmul(k_ose.detach(), key_prototypes.t()) / max(
            float(tau_t), 1e-12)
        p_q = torch.softmax(p_q_logits, dim=1)
        p_k = torch.softmax(p_k_logits, dim=1).detach()
        align_loss = self._soft_cross_entropy(p_q_logits, p_k)

        # Key prototypes are gradient-free. A student copy with the same key-neighbor
        # indices keeps prototype dispersion trainable without changing the teacher.
        student_exemplar_z = self._student_exemplar_features(exemplar)
        student_prototypes, _ = self._class_prototypes(
            student_exemplar_z, nn_idx=nn_idx)
        proto_sim = torch.matmul(
            student_prototypes, student_prototypes.t()) / max(float(tau_s), 1e-12)
        off_diag = ~torch.eye(
            student_prototypes.size(0), dtype=torch.bool,
            device=student_prototypes.device)
        disp_loss = (proto_sim[off_diag].mean()
                     if student_prototypes.size(0) > 1
                     else proto_sim.new_tensor(0.0))

        p_mix_logits = torch.matmul(z_mix, key_prototypes.t()) / max(
            float(tau_s), 1e-12)
        beta = float(mix_beta)
        mix_proto_target = (
            beta * p_q.detach() + (1.0 - beta) * p_k[mix_index].detach())
        mix_proto_loss = self._soft_cross_entropy(
            p_mix_logits, mix_proto_target)

        ins_logits = torch.matmul(z_mix, k_ose.detach().t()) / max(
            float(tau_s), 1e-12)
        log_prob = F.log_softmax(ins_logits, dim=1)
        row = torch.arange(z_mix.size(0), device=z_mix.device)
        mix_ins_loss = -(
            beta * log_prob[row, row] +
            (1.0 - beta) * log_prob[row, mix_index]
        ).mean()

        return {
            'align': align_loss,
            'disp': disp_loss,
            'mix_proto': mix_proto_loss,
            'mix_ins': mix_ins_loss,
            'proto': align_loss + disp_loss,
            'mix': mix_proto_loss + mix_ins_loss,
        }

    def forward(self, im_q_extreme, im_q, im_k=None, nnm=False, topk=1,
                exemplar=None, im_mix=None, mix_index=None, mix_beta=None,
                compute_ose=False, ose_topk=8, ose_alpha=0.75,
                ose_tau_s=0.04, ose_tau_t=0.1):
        if not self.pretrain:
            return self.encoder_q(im_q)

        q_features = self.encoder_q.forward_features(im_q)
        q = F.normalize(self.encoder_q.fc(q_features), dim=1)
        q_extreme_features, q_extreme_drop_features = (
            self.encoder_q.forward_features(im_q_extreme, drop=True))
        q_extreme = F.normalize(
            self.encoder_q.fc(q_extreme_features), dim=1)
        q_extreme_drop = F.normalize(
            self.encoder_q.fc(q_extreme_drop_features), dim=1)

        with torch.no_grad():
            self._momentum_update_key_encoder()
            k_features = self.encoder_k.forward_features(im_k)
            k = F.normalize(self.encoder_k.fc(k_features), dim=1)
            k_ose = (F.normalize(self.ose_projector_k(k_features), dim=1)
                     if compute_ose else None)

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        l_pos_e = torch.einsum('nc,nc->n', [q_extreme, k]).unsqueeze(-1)
        l_neg_e = torch.einsum(
            'nc,ck->nk', [q_extreme, self.queue.clone().detach()])
        l_pos_ed = torch.einsum(
            'nc,nc->n', [q_extreme_drop, k]).unsqueeze(-1)
        l_neg_ed = torch.einsum(
            'nc,ck->nk', [q_extreme_drop, self.queue.clone().detach()])

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
            contrast_target = torch.cat([first_pos, topk_onehot], dim=1)
        else:
            contrast_target = torch.zeros(
                logits.shape[0], dtype=torch.long, device=logits.device)

        ose_losses = None
        if compute_ose:
            if not bool(self.ose_initialized.item()):
                raise RuntimeError('activate_ose() must be called after warmup')
            if exemplar is None or im_mix is None or mix_index is None or mix_beta is None:
                raise ValueError(
                    'OSE losses require exemplar, im_mix, mix_index, and mix_beta')
            q_ose = F.normalize(self.ose_projector_q(q_features), dim=1)
            mix_features = self.encoder_q.forward_features(im_mix)
            z_mix = F.normalize(self.ose_projector_q(mix_features), dim=1)
            ose_losses = self._ose_losses(
                q_ose=q_ose, k_ose=k_ose, exemplar=exemplar, z_mix=z_mix,
                mix_index=mix_index, mix_beta=mix_beta, topk=ose_topk,
                alpha=ose_alpha, tau_s=ose_tau_s, tau_t=ose_tau_t)
            self._dequeue_and_enqueue_ose(k_ose)

        self._dequeue_and_enqueue(k)
        return (logits, contrast_target, logits_e_prob, logits_ed_prob,
                labels_ddm, ose_losses)
