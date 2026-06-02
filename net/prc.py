import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlight import import_class


class PRC(nn.Module):
    def __init__(self, base_encoder=None, feature_dim=128, mlp=True, in_channels=3,
                 hidden_channels=64, hidden_dim=256, dropout=0.5,
                 graph_args={'layout': 'ntu-rgb+d', 'strategy': 'spatial'},
                 edge_importance_weighting=True, **kwargs):
        super().__init__()
        base_encoder = import_class(base_encoder)
        self.encoder = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                    hidden_dim=hidden_dim, num_class=feature_dim,
                                    dropout=dropout, graph_args=graph_args,
                                    edge_importance_weighting=edge_importance_weighting,
                                    **kwargs)
        if mlp:
            dim_mlp = self.encoder.fc.weight.shape[1]
            self.encoder.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                            nn.ReLU(inplace=True),
                                            self.encoder.fc)
        self.feature_dim = int(feature_dim)
        self.heads = nn.ModuleDict()
        self.soft_root_id = None
        self.soft_internal_ids = []
        self.soft_children = {}
        self.soft_leaf_ids = []
        self.soft_leaf_paths = []
        self.leaf_prototypes = nn.ParameterDict()
        self.next_soft_node_id = 0

    def add_split_head(self, node_id):
        key = str(int(node_id))
        if key not in self.heads:
            self.heads[key] = nn.Linear(self.feature_dim, 2)
        return self.heads[key]

    def _prototype_key(self, node_id):
        return str(int(node_id))

    def init_soft_tree(self):
        """Initialize a growable soft tree with one routing root and two leaves."""
        if self.soft_root_id is not None:
            return [], []

        self.soft_root_id = 0
        self.next_soft_node_id = 1
        root_head = self.add_split_head(self.soft_root_id)
        left_id = self._add_soft_leaf()
        right_id = self._add_soft_leaf()
        self.soft_internal_ids = [self.soft_root_id]
        self.soft_children[self.soft_root_id] = (left_id, right_id)
        self._refresh_soft_paths()
        return [root_head], [self.leaf_prototypes[self._prototype_key(left_id)],
                             self.leaf_prototypes[self._prototype_key(right_id)]]

    def _add_soft_leaf(self, prototype=None):
        node_id = self.next_soft_node_id
        self.next_soft_node_id += 1
        if prototype is None:
            prototype = torch.randn(self.feature_dim)
        self.leaf_prototypes[self._prototype_key(node_id)] = nn.Parameter(prototype.clone())
        return node_id

    def _refresh_soft_paths(self):
        leaf_ids = []
        leaf_paths = []

        def visit(node_id, path):
            if node_id not in self.soft_children:
                leaf_ids.append(node_id)
                leaf_paths.append(path)
                return
            left_id, right_id = self.soft_children[node_id]
            visit(left_id, path + [(node_id, 0)])
            visit(right_id, path + [(node_id, 1)])

        if self.soft_root_id is not None:
            visit(self.soft_root_id, [])
        self.soft_leaf_ids = leaf_ids
        self.soft_leaf_paths = leaf_paths

    def grow_soft_tree(self, leaf_ids, noise_scale=0.01):
        """Turn selected leaves into internal routing nodes with two new leaves."""
        new_modules = []
        new_params = []
        for leaf_id in leaf_ids:
            leaf_id = int(leaf_id)
            key = self._prototype_key(leaf_id)
            if key not in self.leaf_prototypes:
                continue

            old_proto = self.leaf_prototypes[key].data.clone()
            del self.leaf_prototypes[key]

            head = self.add_split_head(leaf_id)
            if leaf_id not in self.soft_internal_ids:
                self.soft_internal_ids.append(leaf_id)
            noise = torch.randn_like(old_proto) * float(noise_scale)
            left_id = self._add_soft_leaf(old_proto + noise)
            right_id = self._add_soft_leaf(old_proto - noise)
            self.soft_children[leaf_id] = (left_id, right_id)
            new_modules.append(head)
            new_params.append(self.leaf_prototypes[self._prototype_key(left_id)])
            new_params.append(self.leaf_prototypes[self._prototype_key(right_id)])

        self._refresh_soft_paths()
        return new_modules, new_params

    def forward_soft(self, x, temperature=1.0):
        z = self.forward_features(x)
        if self.soft_root_id is None or not self.soft_leaf_ids:
            raise ValueError('init_soft_tree() must be called before forward_soft')

        route_probs = {}
        for node_id in self.soft_internal_ids:
            logits = self.heads[str(node_id)](z)
            route_probs[node_id] = torch.softmax(logits / max(float(temperature), 1e-6), dim=1)

        reach_probs = {}
        leaf_probs = []

        def visit(node_id, reach):
            if node_id not in self.soft_children:
                leaf_probs.append(reach)
                return
            reach_probs[node_id] = reach
            left_id, right_id = self.soft_children[node_id]
            visit(left_id, reach * route_probs[node_id][:, 0])
            visit(right_id, reach * route_probs[node_id][:, 1])

        visit(self.soft_root_id, z.new_ones(z.size(0)))
        leaf_probs = torch.stack(leaf_probs, dim=1)
        leaf_probs = leaf_probs / leaf_probs.sum(dim=1, keepdim=True).clamp_min(1e-12)

        prototypes = torch.stack(
            [self.leaf_prototypes[self._prototype_key(leaf_id)] for leaf_id in self.soft_leaf_ids], dim=0)
        prototypes = F.normalize(prototypes, dim=1)
        return {
            'features': z,
            'leaf_probs': leaf_probs,
            'route_probs': route_probs,
            'reach_probs': reach_probs,
            'prototypes': prototypes,
            'leaf_ids': list(self.soft_leaf_ids),
        }

    def forward_features(self, x, normalize=True):
        z = self.encoder(x)
        if normalize:
            z = F.normalize(z, dim=1)
        return z

    def forward(self, x, node_ids=None):
        z = self.forward_features(x)
        if node_ids is None:
            node_ids = [int(k) for k in self.heads.keys()]
        logits = {}
        for node_id in node_ids:
            key = str(int(node_id))
            if key in self.heads:
                logits[int(node_id)] = self.heads[key](z)
        return logits, z
