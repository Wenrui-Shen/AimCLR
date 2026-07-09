import torch
import torch.nn.functional as F

from .aimclr import AimCLR


class OSEAimCLR(AimCLR):
    """AimCLR with one-shot exemplar-guided prototype losses."""

    @staticmethod
    def _soft_cross_entropy(logits, target):
        return -torch.mean(torch.sum(F.log_softmax(logits, dim=1) * target, dim=1))

    def _class_prototypes(self, exemplar_z, topk=8, alpha=0.75):
        memory = self.queue.clone().detach().t()
        memory = F.normalize(memory, dim=1)
        exemplar_z = F.normalize(exemplar_z, dim=1)

        sim = torch.matmul(exemplar_z, memory.t())
        if exemplar_z.size(0) > 1:
            other_sim = sim.unsqueeze(0).expand(exemplar_z.size(0), -1, -1).clone()
            eye = torch.eye(exemplar_z.size(0), dtype=torch.bool, device=sim.device)
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
            components = torch.cat([exemplar_z[class_idx:class_idx + 1], neighbors], dim=0)
            weights = torch.softmax(torch.matmul(components, exemplar_z[class_idx]), dim=0)
            prototype = torch.sum(weights.unsqueeze(1) * components, dim=0)
            prototypes.append(prototype)
        prototypes = torch.stack(prototypes, dim=0)
        return F.normalize(prototypes, dim=1)

    def _ose_losses(self, q, k, exemplar, im_mix, mix_index, mix_beta,
                    topk=8, alpha=0.75, tau_s=0.04, tau_t=0.1):
        exemplar_z = self.encoder_q(exemplar)
        exemplar_z = F.normalize(exemplar_z, dim=1)
        prototypes = self._class_prototypes(exemplar_z, topk=topk, alpha=alpha)

        p_q_logits = torch.matmul(q, prototypes.t()) / max(float(tau_s), 1e-12)
        p_k_logits = torch.matmul(k.detach(), prototypes.detach().t()) / max(float(tau_t), 1e-12)
        p_q = torch.softmax(p_q_logits, dim=1)
        p_k = torch.softmax(p_k_logits, dim=1).detach()
        align_loss = self._soft_cross_entropy(p_q_logits, p_k)

        proto_sim = torch.matmul(prototypes, prototypes.t()) / max(float(tau_s), 1e-12)
        off_diag = ~torch.eye(prototypes.size(0), dtype=torch.bool, device=prototypes.device)
        disp_loss = proto_sim[off_diag].mean() if prototypes.size(0) > 1 else proto_sim.new_tensor(0.0)

        z_mix = self.encoder_q(im_mix)
        z_mix = F.normalize(z_mix, dim=1)
        p_mix_logits = torch.matmul(z_mix, prototypes.t()) / max(float(tau_s), 1e-12)
        beta = float(mix_beta)
        mix_proto_target = beta * p_q.detach() + (1.0 - beta) * p_k[mix_index].detach()
        mix_proto_loss = self._soft_cross_entropy(p_mix_logits, mix_proto_target)

        ins_logits = torch.matmul(z_mix, k.detach().t()) / max(float(tau_s), 1e-12)
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

        q = self.encoder_q(im_q)
        q_extreme, q_extreme_drop = self.encoder_q(im_q_extreme, drop=True)
        q = F.normalize(q, dim=1)
        q_extreme = F.normalize(q_extreme, dim=1)
        q_extreme_drop = F.normalize(q_extreme_drop, dim=1)

        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = self.encoder_k(im_k)
            k = F.normalize(k, dim=1)

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        l_pos_e = torch.einsum('nc,nc->n', [q_extreme, k]).unsqueeze(-1)
        l_neg_e = torch.einsum('nc,ck->nk', [q_extreme, self.queue.clone().detach()])
        l_pos_ed = torch.einsum('nc,nc->n', [q_extreme_drop, k]).unsqueeze(-1)
        l_neg_ed = torch.einsum('nc,ck->nk', [q_extreme_drop, self.queue.clone().detach()])

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
            first_pos = torch.ones(topk_onehot.size(0), 1, device=topk_onehot.device)
            contrast_target = torch.cat([first_pos, topk_onehot], dim=1)
        else:
            contrast_target = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

        ose_losses = None
        if compute_ose:
            if exemplar is None or im_mix is None or mix_index is None or mix_beta is None:
                raise ValueError('OSE losses require exemplar, im_mix, mix_index, and mix_beta')
            ose_losses = self._ose_losses(
                q=q, k=k, exemplar=exemplar, im_mix=im_mix, mix_index=mix_index,
                mix_beta=mix_beta, topk=ose_topk, alpha=ose_alpha,
                tau_s=ose_tau_s, tau_t=ose_tau_t)

        self._dequeue_and_enqueue(k)

        return logits, contrast_target, logits_e_prob, logits_ed_prob, labels_ddm, ose_losses
