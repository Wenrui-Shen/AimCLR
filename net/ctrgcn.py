"""CTR-GCN backbone adapted for the AimCLR model interface.

Architecture adapted from the official CTR-GCN implementation:
https://github.com/Uason-Chen/CTR-GCN

The upstream implementation is licensed under CC BY-NC 4.0. Changes here
provide repository-compatible constructor arguments, ``forward_features()``,
device-safe fixed adjacency handling, and a reset hook for official-style
initialization.
"""

import math

import numpy as np
import torch
import torch.nn as nn


_NUM_POINT = 25
_SELF_LINK = [(index, index) for index in range(_NUM_POINT)]
_INWARD_ORIGINAL = [
    (1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
    (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
    (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
    (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
    (22, 23), (23, 8), (24, 25), (25, 12),
]
_INWARD = [(source - 1, target - 1)
           for source, target in _INWARD_ORIGINAL]
_OUTWARD = [(target, source) for source, target in _INWARD]


def _edge_to_matrix(edges, num_point):
    matrix = np.zeros((num_point, num_point), dtype=np.float32)
    for source, target in edges:
        matrix[target, source] = 1.0
    return matrix


def _normalize_digraph(matrix):
    degree = np.sum(matrix, axis=0)
    inverse_degree = np.zeros_like(degree)
    nonzero = degree > 0
    inverse_degree[nonzero] = degree[nonzero] ** -1
    return np.dot(matrix, np.diag(inverse_degree)).astype(np.float32)


def _ntu_spatial_adjacency():
    identity = _edge_to_matrix(_SELF_LINK, _NUM_POINT)
    inward = _normalize_digraph(_edge_to_matrix(_INWARD, _NUM_POINT))
    outward = _normalize_digraph(_edge_to_matrix(_OUTWARD, _NUM_POINT))
    return np.stack((identity, inward, outward)).astype(np.float32)


def _conv_init(module):
    if module.weight is not None:
        nn.init.kaiming_normal_(module.weight, mode='fan_out')
    if module.bias is not None:
        nn.init.constant_(module.bias, 0)


def _bn_init(module, scale):
    nn.init.constant_(module.weight, scale)
    nn.init.constant_(module.bias, 0)


def _branch_weights_init(module):
    if isinstance(module, nn.Conv2d):
        _conv_init(module)
    elif isinstance(module, nn.BatchNorm2d):
        module.weight.data.normal_(1.0, 0.02)
        module.bias.data.zero_()


class TemporalConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, dilation=1):
        super().__init__()
        padding = (
            kernel_size + (kernel_size - 1) * (dilation - 1) - 1
        ) // 2
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=(kernel_size, 1),
            padding=(padding, 0),
            stride=(stride, 1),
            dilation=(dilation, 1))
        self.bn = nn.BatchNorm2d(out_channels)

    def reset_parameters(self):
        _conv_init(self.conv)
        _bn_init(self.bn, 1.0)

    def forward(self, x):
        return self.bn(self.conv(x))


class MultiScaleTemporalConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=5,
                 stride=1, dilations=(1, 2), residual=True):
        super().__init__()
        num_branches = len(dilations) + 2
        if out_channels % num_branches != 0:
            raise ValueError(
                'CTR-GCN output channels must be divisible by {}'.format(
                    num_branches))
        branch_channels = out_channels // num_branches
        kernel_sizes = (
            list(kernel_size)
            if isinstance(kernel_size, (tuple, list))
            else [kernel_size] * len(dilations))
        if len(kernel_sizes) != len(dilations):
            raise ValueError('Temporal kernels and dilations must align')

        branches = []
        for branch_kernel, dilation in zip(kernel_sizes, dilations):
            branches.append(nn.Sequential(
                nn.Conv2d(in_channels, branch_channels, kernel_size=1),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                TemporalConv(
                    branch_channels, branch_channels,
                    kernel_size=branch_kernel,
                    stride=stride, dilation=dilation),
            ))
        branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(
                kernel_size=(3, 1), stride=(stride, 1),
                padding=(1, 0)),
            nn.BatchNorm2d(branch_channels),
        ))
        branches.append(nn.Sequential(
            nn.Conv2d(
                in_channels, branch_channels, kernel_size=1,
                stride=(stride, 1)),
            nn.BatchNorm2d(branch_channels),
        ))
        self.branches = nn.ModuleList(branches)

        if not residual:
            self.residual = lambda value: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda value: value
        else:
            self.residual = TemporalConv(
                in_channels, out_channels, kernel_size=1, stride=stride)

    def reset_parameters(self):
        self.apply(_branch_weights_init)

    def forward(self, x):
        residual = self.residual(x)
        output = torch.cat(
            [branch(x) for branch in self.branches], dim=1)
        return output + residual


class ChannelTopologyRefinement(nn.Module):

    def __init__(self, in_channels, out_channels,
                 relation_reduction=8, middle_reduction=1):
        super().__init__()
        if in_channels in (3, 9):
            relation_channels = 8
        else:
            relation_channels = in_channels // relation_reduction
        if relation_channels < 1 or in_channels // middle_reduction < 1:
            raise ValueError('CTR-GC channel reduction produced zero channels')

        self.conv1 = nn.Conv2d(
            in_channels, relation_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(
            in_channels, relation_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(
            in_channels, out_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(
            relation_channels, out_channels, kernel_size=1)
        self.tanh = nn.Tanh()

    def reset_parameters(self):
        for module in (self.conv1, self.conv2, self.conv3, self.conv4):
            _conv_init(module)

    def forward(self, x, adjacency=None, alpha=1.0):
        source = self.conv1(x).mean(-2)
        target = self.conv2(x).mean(-2)
        features = self.conv3(x)
        relation = self.tanh(
            source.unsqueeze(-1) - target.unsqueeze(-2))
        relation = self.conv4(relation) * alpha
        if adjacency is not None:
            relation = relation + adjacency.unsqueeze(0).unsqueeze(0)
        return torch.einsum('ncuv,nctv->nctu', relation, features)


class UnitGCN(nn.Module):

    def __init__(self, in_channels, out_channels, adjacency,
                 adaptive=True, residual=True):
        super().__init__()
        self.adaptive = bool(adaptive)
        self.num_subset = adjacency.shape[0]
        self.convs = nn.ModuleList([
            ChannelTopologyRefinement(in_channels, out_channels)
            for _ in range(self.num_subset)
        ])
        if residual:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1),
                    nn.BatchNorm2d(out_channels))
            else:
                self.down = lambda value: value
        else:
            self.down = lambda value: 0

        adjacency_tensor = torch.from_numpy(
            adjacency.astype(np.float32))
        if self.adaptive:
            self.PA = nn.Parameter(adjacency_tensor)
        else:
            self.register_buffer('A', adjacency_tensor)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def reset_parameters(self):
        for convolution in self.convs:
            convolution.reset_parameters()
        if isinstance(self.down, nn.Module):
            for module in self.down.modules():
                if isinstance(module, nn.Conv2d):
                    _conv_init(module)
                elif isinstance(module, nn.BatchNorm2d):
                    _bn_init(module, 1.0)
        _bn_init(self.bn, 1e-6)
        nn.init.constant_(self.alpha, 0)
        if self.adaptive:
            self.PA.data.copy_(
                torch.from_numpy(_ntu_spatial_adjacency()).to(
                    self.PA.device))

    def forward(self, x):
        adjacency = self.PA if self.adaptive else self.A
        output = None
        for subset_index, convolution in enumerate(self.convs):
            subset_output = convolution(
                x, adjacency[subset_index], self.alpha)
            output = (
                subset_output if output is None
                else output + subset_output)
        output = self.bn(output)
        output = output + self.down(x)
        return self.relu(output)


class TCNGCNUnit(nn.Module):

    def __init__(self, in_channels, out_channels, adjacency,
                 stride=1, residual=True, adaptive=True):
        super().__init__()
        self.gcn = UnitGCN(
            in_channels, out_channels, adjacency,
            adaptive=adaptive)
        self.tcn = MultiScaleTemporalConv(
            out_channels, out_channels,
            kernel_size=5, stride=stride,
            dilations=(1, 2), residual=False)
        if not residual:
            self.residual = lambda value: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda value: value
        else:
            self.residual = TemporalConv(
                in_channels, out_channels,
                kernel_size=1, stride=stride)
        self.relu = nn.ReLU(inplace=True)

    def reset_parameters(self):
        self.gcn.reset_parameters()
        self.tcn.reset_parameters()
        if isinstance(self.residual, TemporalConv):
            self.residual.reset_parameters()

    def forward(self, x):
        return self.relu(
            self.tcn(self.gcn(x)) + self.residual(x))


class Model(nn.Module):
    """Official-width CTR-GCN for NTU RGB+D skeleton tensors."""

    def __init__(self, in_channels=3, hidden_channels=64,
                 hidden_dim=256, num_class=60, dropout=0.0,
                 graph_args=None, edge_importance_weighting=True,
                 num_point=25, num_person=2, adaptive=True, **kwargs):
        super().__init__()
        del edge_importance_weighting, kwargs
        graph_args = graph_args or {}
        layout = graph_args.get('layout', 'ntu-rgb+d')
        strategy = graph_args.get(
            'strategy', graph_args.get('labeling_mode', 'spatial'))
        if layout not in ('ntu-rgb+d', 'ntu'):
            raise ValueError(
                'CTR-GCN currently supports only NTU RGB+D layout')
        if strategy != 'spatial':
            raise ValueError(
                'CTR-GCN requires the spatial adjacency strategy')
        if int(num_point) != _NUM_POINT:
            raise ValueError('CTR-GCN NTU layout requires 25 joints')
        if int(hidden_dim) != int(hidden_channels) * 4:
            raise ValueError(
                'CTR-GCN hidden_dim must equal 4 * hidden_channels')

        self.num_point = int(num_point)
        self.num_person = int(num_person)
        self.in_channels = int(in_channels)
        base_channels = int(hidden_channels)
        output_channels = int(hidden_dim)
        adjacency = _ntu_spatial_adjacency()

        self.data_bn = nn.BatchNorm1d(
            self.num_person * self.in_channels * self.num_point)
        self.layers = nn.ModuleList((
            TCNGCNUnit(
                self.in_channels, base_channels, adjacency,
                residual=False, adaptive=adaptive),
            TCNGCNUnit(
                base_channels, base_channels, adjacency,
                adaptive=adaptive),
            TCNGCNUnit(
                base_channels, base_channels, adjacency,
                adaptive=adaptive),
            TCNGCNUnit(
                base_channels, base_channels, adjacency,
                adaptive=adaptive),
            TCNGCNUnit(
                base_channels, base_channels * 2, adjacency,
                stride=2, adaptive=adaptive),
            TCNGCNUnit(
                base_channels * 2, base_channels * 2, adjacency,
                adaptive=adaptive),
            TCNGCNUnit(
                base_channels * 2, base_channels * 2, adjacency,
                adaptive=adaptive),
            TCNGCNUnit(
                base_channels * 2, output_channels, adjacency,
                stride=2, adaptive=adaptive),
            TCNGCNUnit(
                output_channels, output_channels, adjacency,
                adaptive=adaptive),
            TCNGCNUnit(
                output_channels, output_channels, adjacency,
                adaptive=adaptive),
        ))
        self.dropout = (
            nn.Dropout(float(dropout))
            if float(dropout) > 0 else nn.Identity())
        self.fc = nn.Linear(output_channels, int(num_class))
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        _bn_init(self.data_bn, 1.0)
        nn.init.normal_(
            self.fc.weight, 0,
            math.sqrt(2.0 / self.fc.out_features))
        if self.fc.bias is not None:
            nn.init.constant_(self.fc.bias, 0)

    def forward_features(self, x):
        if x.dim() == 3:
            batch_size, time_steps, flattened = x.shape
            expected = self.num_point * self.in_channels
            if flattened != expected:
                raise ValueError(
                    'Flattened CTR-GCN input has {} channels, expected {}'
                    .format(flattened, expected))
            x = x.view(
                batch_size, time_steps, self.num_point,
                self.in_channels)
            x = x.permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        if x.dim() != 5:
            raise ValueError(
                'CTR-GCN expects input shaped [N, C, T, V, M]')

        batch_size, channels, time_steps, num_point, num_person = x.size()
        if channels != self.in_channels:
            raise ValueError('Unexpected number of input channels')
        if num_point != self.num_point:
            raise ValueError('Unexpected number of skeleton joints')
        if num_person != self.num_person:
            raise ValueError(
                'CTR-GCN was configured for {} persons, received {}'
                .format(self.num_person, num_person))

        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(
            batch_size,
            num_person * num_point * channels,
            time_steps)
        x = self.data_bn(x)
        x = x.view(
            batch_size, num_person, num_point,
            channels, time_steps)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(
            batch_size * num_person,
            channels, time_steps, num_point)

        for layer in self.layers:
            x = layer(x)

        output_channels = x.size(1)
        x = x.view(batch_size, num_person, output_channels, -1)
        x = x.mean(dim=3).mean(dim=1)
        return self.dropout(x)

    def forward(self, x):
        return self.fc(self.forward_features(x))
