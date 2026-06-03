"""SNN architectures used for motor-imagery EEG classification."""

import torch
import torch.nn as nn
from tools.neurons import CUPYPLIFNode, CUPYLIFNode
from tools import functional, surrogate
import math


class ALIFNode(nn.Module):
    def __init__(self, tau_mem=2.0, tau_adp=20.0, v_threshold=1.0, v_reset=0.0,
                 adapt_scale=1.0, surrogate_function=surrogate.Sigmoid(), detach_reset=True):
        super(ALIFNode, self).__init__()
        if tau_mem <= 0 or tau_adp <= 0:
            raise ValueError('tau_mem and tau_adp must be positive.')
        self.decay_mem = math.exp(-1.0 / tau_mem)
        self.decay_adp = math.exp(-1.0 / tau_adp)
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.adapt_scale = adapt_scale
        self.surrogate_function = surrogate_function
        self.detach_reset = detach_reset
        self.step_mode = 'm'
        self.v = None
        self.a = None

    def reset(self):
        self.v = None
        self.a = None

    def _init_state(self, x):
        if self.v is None or self.v.shape != x.shape:
            self.v = torch.zeros_like(x)
            self.a = torch.zeros_like(x)

    def single_step_forward(self, x):
        self._init_state(x)
        adaptive_threshold = self.v_threshold + self.adapt_scale * self.a
        self.v = self.decay_mem * self.v + (1.0 - self.decay_mem) * x
        spike = self.surrogate_function(self.v - adaptive_threshold)
        spike_for_reset = spike.detach() if self.detach_reset else spike
        self.v = self.v * (1.0 - spike_for_reset) + self.v_reset * spike_for_reset
        self.a = self.decay_adp * self.a + (1.0 - self.decay_adp) * spike
        return spike

    def multi_step_forward(self, x_seq):
        out_seq = []
        for t in range(x_seq.shape[0]):
            out_seq.append(self.single_step_forward(x_seq[t]))
        return torch.stack(out_seq, dim=0)

    def forward(self, x):
        if self.step_mode == 's':
            return self.single_step_forward(x)
        if self.step_mode == 'm':
            return self.multi_step_forward(x)
        raise ValueError(f'Unsupported step_mode: {self.step_mode}')


class CUPY_SNN_2ALIF(nn.Module):
    def __init__(self, in_channels=22, out_num=4, beta=2, w=0.5, surrogate_function=surrogate.Sigmoid(),
                 time_step=128 * 3, tau_adp_scale=10.0, adapt_scale=1.0):
        super(CUPY_SNN_2ALIF, self).__init__()
        tau_mem = math.exp(-w) + 1
        tau_adp = tau_mem * tau_adp_scale
        channels = int(beta * in_channels)
        kernal = time_step // 32
        self.encode_C = nn.Conv1d(in_channels, channels, kernel_size=(1,), bias=False)
        self.encode_T = nn.Conv1d(channels, channels, kernel_size=(kernal,), padding=(kernal // 2,),
                                  groups=channels, bias=False)
        self.bn_T = nn.BatchNorm1d(channels)
        self.neuron1 = ALIFNode(tau_mem=tau_mem, tau_adp=tau_adp, adapt_scale=adapt_scale,
                                surrogate_function=surrogate_function)
        self.neuron2 = ALIFNode(tau_mem=tau_mem, tau_adp=tau_adp, adapt_scale=adapt_scale,
                                surrogate_function=surrogate_function)
        self.Classify = nn.Linear(in_features=channels, out_features=out_num)
        functional.set_step_mode(self, step_mode='m')

    def forward(self, x):
        x = self.encode_C(x)
        x = self.encode_T(x)
        x = self.bn_T(x).permute(2, 0, 1)
        x = self.neuron1(x)
        x = self.neuron2(x)
        x = x.mean(0)
        x = self.Classify(x)
        return x


class CUPY_SNN_2PLIF(nn.Module):
    def __init__(self, in_channels=22, out_num=4, beta=2, w=0.5, surrogate_function=surrogate.Sigmoid(), time_step=128 * 3):
        super(CUPY_SNN_2PLIF, self).__init__()
        tau = math.exp(-w) + 1
        channels = int(beta * in_channels)
        kernal = time_step // 32
        self.encode_C = nn.Conv1d(in_channels, channels, kernel_size=(1,), bias=False)
        self.encode_T = nn.Conv1d(channels, channels, kernel_size=(kernal,), padding=(kernal // 2,),
                                  groups=channels, bias=False)
        self.bn_T = nn.BatchNorm1d(channels)
        self.neuron1 = CUPYPLIFNode(init_tau=tau, surrogate_function=surrogate_function)
        self.neuron2 = CUPYPLIFNode(init_tau=tau, surrogate_function=surrogate_function)
        self.Classify = nn.Linear(in_features=channels, out_features=out_num)
        functional.set_step_mode(self, step_mode='m')

    def forward(self, x):
        x = self.encode_C(x)
        x = self.encode_T(x)
        x = self.bn_T(x).permute(2, 0, 1)
        x = self.neuron1(x)
        x = self.neuron2(x)
        x = x.mean(0)
        x = self.Classify(x)
        return x

class CUPY_SNN_PLIF(nn.Module):
    def __init__(self, in_channels=22, out_num=4, beta=2, w=0.5, surrogate_function=surrogate.Sigmoid(), time_step=128 * 3):
        super(CUPY_SNN_PLIF, self).__init__()
        tau = math.exp(-w) + 1
        channels = int(beta * in_channels)
        kernal = time_step // 32
        self.encode_C = nn.Conv1d(in_channels, channels, kernel_size=(1,), bias=False)
        self.encode_T = nn.Conv1d(channels, channels, kernel_size=(kernal,), padding=(kernal // 2,),
                                  groups=channels, bias=False)
        self.bn_T = nn.BatchNorm1d(channels)
        self.neuron = CUPYPLIFNode(init_tau=tau, surrogate_function=surrogate_function)
        self.Classify = nn.Linear(in_features=channels, out_features=out_num)
        functional.set_step_mode(self, step_mode='m')

    def forward(self, x):
        x = self.encode_C(x)
        x = self.encode_T(x)
        x = self.bn_T(x).permute(2, 0, 1)
        x = self.neuron(x)
        x = x.mean(0)
        x = self.Classify(x)
        return x


class CUPY_SNN_3PLIF_PARALLEL(nn.Module):
    def __init__(self, in_channels=22, out_num=4, beta=2, w=0.5, surrogate_function=surrogate.Sigmoid(),
                 time_step=128 * 3, head_tau_spread=0.2, head_vth_spread=0.1, head_dropout=0.1):
        super(CUPY_SNN_3PLIF_PARALLEL, self).__init__()
        tau = math.exp(-w) + 1
        channels = int(beta * in_channels)
        kernal = time_step // 32
        self.encode_C = nn.Conv1d(in_channels, channels, kernel_size=(1,), bias=False)
        self.encode_T = nn.Conv1d(channels, channels, kernel_size=(kernal,), padding=(kernal // 2,),
                                  groups=channels, bias=False)
        self.bn_T = nn.BatchNorm1d(channels)

        tau_factors = (1.0 - head_tau_spread, 1.0, 1.0 + head_tau_spread)
        v_thresholds = (1.0 - head_vth_spread, 1.0, 1.0 + head_vth_spread)
        self.neuron_head1 = CUPYPLIFNode(init_tau=tau * tau_factors[0], v_threshold=v_thresholds[0], surrogate_function=surrogate_function)
        self.neuron_head2 = CUPYPLIFNode(init_tau=tau * tau_factors[1], v_threshold=v_thresholds[1], surrogate_function=surrogate_function)
        self.neuron_head3 = CUPYPLIFNode(init_tau=tau * tau_factors[2], v_threshold=v_thresholds[2], surrogate_function=surrogate_function)
        self.head_dropout = nn.Dropout(p=head_dropout)
        self.head_merge_logits = nn.Parameter(torch.zeros(3))

        self.Classify = nn.Linear(in_features=channels * 3, out_features=out_num)
        functional.set_step_mode(self, step_mode='m')

    def forward(self, x):
        x = self.encode_C(x)
        x = self.encode_T(x)
        x = self.bn_T(x).permute(2, 0, 1)

        x1 = self.neuron_head1(x)
        x2 = self.neuron_head2(x)
        x3 = self.neuron_head3(x)
        x = torch.stack((x1, x2, x3), dim=2)
        head_merge = torch.softmax(self.head_merge_logits, dim=0).view(1, 1, 3, 1)
        x = self.head_dropout(x * head_merge)
        x = x.flatten(2)

        x = x.mean(0)
        x = self.Classify(x)
        return x


class CUPY_SNN_3PLIF_PARALLEL_ALIF_READOUT(nn.Module):
    def __init__(self, in_channels=22, out_num=4, beta=2, w=0.5, surrogate_function=surrogate.Sigmoid(),
                 time_step=128 * 3, readout_adapt_scale=0.02, readout_tau_adp_scale=6.0,
                 readout_v_threshold=0.2, readout_input_scale=2.5, head_tau_spread=0.2,
                 head_vth_spread=0.1, head_dropout=0.1):
        super(CUPY_SNN_3PLIF_PARALLEL_ALIF_READOUT, self).__init__()
        tau_mem = math.exp(-w) + 1
        tau_adp = tau_mem * readout_tau_adp_scale
        channels = int(beta * in_channels)
        kernal = time_step // 32
        self.encode_C = nn.Conv1d(in_channels, channels, kernel_size=(1,), bias=False)
        self.encode_T = nn.Conv1d(channels, channels, kernel_size=(kernal,), padding=(kernal // 2,),
                                  groups=channels, bias=False)
        self.bn_T = nn.BatchNorm1d(channels)

        tau_factors = (1.0 - head_tau_spread, 1.0, 1.0 + head_tau_spread)
        v_thresholds = (1.0 - head_vth_spread, 1.0, 1.0 + head_vth_spread)
        self.neuron_head1 = CUPYPLIFNode(init_tau=tau_mem * tau_factors[0], v_threshold=v_thresholds[0], surrogate_function=surrogate_function)
        self.neuron_head2 = CUPYPLIFNode(init_tau=tau_mem * tau_factors[1], v_threshold=v_thresholds[1], surrogate_function=surrogate_function)
        self.neuron_head3 = CUPYPLIFNode(init_tau=tau_mem * tau_factors[2], v_threshold=v_thresholds[2], surrogate_function=surrogate_function)
        self.head_dropout = nn.Dropout(p=head_dropout)
        self.head_merge_logits = nn.Parameter(torch.zeros(3))

        self.fuse = nn.Linear(in_features=channels * 3, out_features=out_num)
        self.bn_readout = nn.BatchNorm1d(out_num)
        self.readout_input_scale = readout_input_scale
        self.readout_neuron = ALIFNode(
            tau_mem=tau_mem,
            tau_adp=tau_adp,
            v_threshold=readout_v_threshold,
            adapt_scale=readout_adapt_scale,
            surrogate_function=surrogate_function,
            detach_reset=True,
        )
        functional.set_step_mode(self, step_mode='m')

    def forward(self, x):
        x = self.encode_C(x)
        x = self.encode_T(x)
        x = self.bn_T(x).permute(2, 0, 1)

        x1 = self.neuron_head1(x)
        x2 = self.neuron_head2(x)
        x3 = self.neuron_head3(x)
        x = torch.stack((x1, x2, x3), dim=2)
        head_merge = torch.softmax(self.head_merge_logits, dim=0).view(1, 1, 3, 1)
        x = self.head_dropout(x * head_merge)
        x = x.flatten(2)

        x = self.fuse(x)
        x = self.bn_readout(x.permute(1, 2, 0)).permute(2, 0, 1)
        x = x * self.readout_input_scale
        x = self.readout_neuron(x)
        x = x.mean(0)
        return x


class CUPY_SNN_ALIF_READOUT(nn.Module):
    def __init__(self, in_channels=22, out_num=4, beta=2, w=0.5, surrogate_function=surrogate.Sigmoid(),
                 time_step=128 * 3, readout_adapt_scale=0.02, readout_tau_adp_scale=6.0,
                 readout_v_threshold=0.2, readout_input_scale=2.5):
        super(CUPY_SNN_ALIF_READOUT, self).__init__()
        tau_mem = math.exp(-w) + 1
        tau_adp = tau_mem * readout_tau_adp_scale
        channels = int(beta * in_channels)
        kernal = time_step // 32
        self.encode_C = nn.Conv1d(in_channels, channels, kernel_size=(1,), bias=False)
        self.encode_T = nn.Conv1d(channels, channels, kernel_size=(kernal,), padding=(kernal // 2,),
                                  groups=channels, bias=False)
        self.bn_T = nn.BatchNorm1d(channels)
        self.neuron = CUPYPLIFNode(init_tau=tau_mem, surrogate_function=surrogate_function)
        self.Classify = nn.Linear(in_features=channels, out_features=out_num)
        self.bn_readout = nn.BatchNorm1d(out_num)
        self.readout_input_scale = readout_input_scale
        self.readout_neuron = ALIFNode(
            tau_mem=tau_mem,
            tau_adp=tau_adp,
            v_threshold=readout_v_threshold,
            adapt_scale=readout_adapt_scale,
            surrogate_function=surrogate_function,
            detach_reset=True,
        )
        functional.set_step_mode(self, step_mode='m')

    def forward(self, x):
        x = self.encode_C(x)
        x = self.encode_T(x)
        x = self.bn_T(x).permute(2, 0, 1)
        x = self.neuron(x)
        x = self.Classify(x)
        x = self.bn_readout(x.permute(1, 2, 0)).permute(2, 0, 1)
        x = x * self.readout_input_scale
        x = self.readout_neuron(x)
        x = x.mean(0)
        return x


class CUPY_SNN_LIF_READOUT(nn.Module):
    def __init__(self, in_channels=22, out_num=4, beta=2, w=0.5, surrogate_function=surrogate.Sigmoid(),
                 time_step=128 * 3, readout_adapt_scale=0.02, readout_tau_adp_scale=6.0,
                 readout_v_threshold=0.2, readout_input_scale=2.5, dropout=0.0):
        super(CUPY_SNN_LIF_READOUT, self).__init__()
        tau_mem = math.exp(-w) + 1
        channels = int(beta * in_channels)
        kernal = time_step // 32
        self.encode_C = nn.Conv1d(in_channels, channels, kernel_size=(1,), bias=False)
        self.encode_T = nn.Conv1d(channels, channels, kernel_size=(kernal,), padding=(kernal // 2,),
                                  groups=channels, bias=False)
        self.bn_T = nn.BatchNorm1d(channels)
        self.neuron = CUPYPLIFNode(init_tau=tau_mem, surrogate_function=surrogate_function)
        self.drop = nn.Dropout(p=dropout) if dropout and dropout > 0 else nn.Identity()
        self.Classify = nn.Linear(in_features=channels, out_features=out_num)
        self.bn_readout = nn.BatchNorm1d(out_num)
        self.readout_input_scale = readout_input_scale
        self.readout_neuron = CUPYLIFNode(
            surrogate_function=surrogate_function,
            v_threshold=readout_v_threshold,
            v_reset=0.0,
            detach_reset=True,
            backend='torch',
            w=w,
        )
        functional.set_step_mode(self, step_mode='m')

    def forward(self, x):
        x = self.encode_C(x)
        x = self.encode_T(x)
        x = self.bn_T(x).permute(2, 0, 1)
        x = self.neuron(x)
        x = self.drop(x)
        x = self.Classify(x)
        x = self.bn_readout(x.permute(1, 2, 0)).permute(2, 0, 1)
        x = x * self.readout_input_scale
        x = self.readout_neuron(x)
        x = x.mean(0)
        return x


class SignedSpikeEncoder(nn.Module):
    def __init__(self, threshold=0.5, scale=1.0, surrogate_function=surrogate.Sigmoid()):
        super(SignedSpikeEncoder, self).__init__()
        self.threshold = float(threshold)
        self.scale = float(scale)
        self.surrogate_function = surrogate_function

    def forward(self, x):
        x = x * self.scale
        pos = self.surrogate_function(x - self.threshold)
        neg = self.surrogate_function(-x - self.threshold)
        return torch.cat((pos, neg), dim=-1)


class CUPY_SNN_SPIKING_CONV_LIF_READOUT(nn.Module):
    """CUPY LIF readout with spike-domain convolutional feature extraction.

    Unlike CUPY_SNN_LIF_READOUT, the EEG stream is first converted to signed
    surrogate spikes. Spatial and temporal convolutions then operate on the
    spike train and are separated by LIF dynamics.
    """

    def __init__(self, in_channels=22, out_num=4, beta=2, w=0.5, surrogate_function=surrogate.Sigmoid(),
                 time_step=128 * 3, readout_adapt_scale=0.02, readout_tau_adp_scale=6.0,
                 readout_v_threshold=0.2, readout_input_scale=2.5, encoder_threshold=0.5,
                 encoder_scale=1.0, dropout=0.1, lif_v_threshold=0.5, lif_input_scale=2.5):
        super(CUPY_SNN_SPIKING_CONV_LIF_READOUT, self).__init__()
        tau_mem = math.exp(-w) + 1
        channels = int(beta * in_channels)
        kernal = time_step // 32

        self.input_norm = nn.BatchNorm1d(in_channels)
        self.spike_encoder = SignedSpikeEncoder(
            threshold=encoder_threshold,
            scale=encoder_scale,
            surrogate_function=surrogate_function,
        )
        self.encode_C = nn.Conv1d(in_channels * 2, channels, kernel_size=(1,), bias=False)
        self.bn_C = nn.BatchNorm1d(channels)
        self.lif_input_scale = lif_input_scale
        self.spatial_lif = CUPYLIFNode(
            surrogate_function=surrogate_function,
            v_threshold=lif_v_threshold,
            v_reset=0.0,
            detach_reset=True,
            backend='torch',
            w=w,
        )
        self.encode_T = nn.Conv1d(channels, channels, kernel_size=(kernal,), padding=(kernal // 2,),
                                  groups=channels, bias=False)
        self.bn_T = nn.BatchNorm1d(channels)
        self.drop_T = nn.Dropout(p=dropout) if dropout and dropout > 0 else nn.Identity()
        self.temporal_lif = CUPYPLIFNode(init_tau=tau_mem, surrogate_function=surrogate_function)
        self.Classify = nn.Linear(in_features=channels, out_features=out_num)
        self.bn_readout = nn.BatchNorm1d(out_num)
        self.readout_input_scale = readout_input_scale
        self.readout_neuron = CUPYLIFNode(
            surrogate_function=surrogate_function,
            v_threshold=readout_v_threshold,
            v_reset=0.0,
            detach_reset=True,
            backend='torch',
            w=w,
        )
        functional.set_step_mode(self, step_mode='m')

    def forward(self, x):
        x = self.input_norm(x)
        x = x.permute(2, 0, 1)
        x = self.spike_encoder(x)
        x = x.permute(1, 2, 0)

        x = self.encode_C(x)
        x = self.bn_C(x).permute(2, 0, 1)
        x = x * self.lif_input_scale
        x = self.spatial_lif(x)

        x = self.encode_T(x.permute(1, 2, 0))
        x = self.bn_T(x)
        x = self.drop_T(x).permute(2, 0, 1)
        x = self.temporal_lif(x)

        x = self.Classify(x)
        x = self.bn_readout(x.permute(1, 2, 0)).permute(2, 0, 1)
        x = x * self.readout_input_scale
        x = self.readout_neuron(x)
        x = x.mean(0)
        return x


class CUPY_SNN_SIGNED_LIF_MLP_READOUT(nn.Module):
    """Channel-mixing SNN without temporal/channel convolutions.

    The continuous EEG sample is normalized over channels and converted to
    signed spikes at each time step. Linear layers then mix channels, while LIF
    cells provide the temporal dynamics and a LIF readout mirrors
    CUPY_SNN_LIF_READOUT.
    """

    def __init__(self, in_channels=22, out_num=4, beta=2, w=0.5, surrogate_function=surrogate.Sigmoid(),
                 time_step=128 * 3, readout_adapt_scale=0.02, readout_tau_adp_scale=6.0,
                 readout_v_threshold=0.2, readout_input_scale=2.5, encoder_threshold=0.5,
                 encoder_scale=1.0, hidden_layers=2, dropout=0.1, lif_v_threshold=0.5):
        super(CUPY_SNN_SIGNED_LIF_MLP_READOUT, self).__init__()
        channels = int(beta * in_channels)
        hidden_layers = max(1, int(hidden_layers))

        self.input_norm = nn.BatchNorm1d(in_channels)
        self.spike_encoder = SignedSpikeEncoder(
            threshold=encoder_threshold,
            scale=encoder_scale,
            surrogate_function=surrogate_function,
        )
        self.proj_in = nn.Linear(in_channels * 2, channels, bias=False)
        self.norm_in = nn.BatchNorm1d(channels)
        self.drop_in = nn.Dropout(p=dropout) if dropout and dropout > 0 else nn.Identity()
        self.lif_in = CUPYLIFNode(
            surrogate_function=surrogate_function,
            v_threshold=lif_v_threshold,
            v_reset=0.0,
            detach_reset=True,
            backend='torch',
            w=w,
        )

        self.projs = nn.ModuleList([
            nn.Linear(channels, channels, bias=False) for _ in range(hidden_layers - 1)
        ])
        self.norms = nn.ModuleList([
            nn.BatchNorm1d(channels) for _ in range(hidden_layers - 1)
        ])
        self.drops = nn.ModuleList([
            nn.Dropout(p=dropout) if dropout and dropout > 0 else nn.Identity()
            for _ in range(hidden_layers - 1)
        ])
        self.lifs = nn.ModuleList([
            CUPYLIFNode(
                surrogate_function=surrogate_function,
                v_threshold=lif_v_threshold,
                v_reset=0.0,
                detach_reset=True,
                backend='torch',
                w=w,
            ) for _ in range(hidden_layers - 1)
        ])

        self.Classify = nn.Linear(in_features=channels, out_features=out_num)
        self.bn_readout = nn.BatchNorm1d(out_num)
        self.readout_input_scale = readout_input_scale
        self.readout_neuron = CUPYLIFNode(
            surrogate_function=surrogate_function,
            v_threshold=readout_v_threshold,
            v_reset=0.0,
            detach_reset=True,
            backend='torch',
            w=w,
        )
        functional.set_step_mode(self, step_mode='m')

    @staticmethod
    def _apply_time_bn(x, bn):
        t, n, c = x.shape
        return bn(x.reshape(t * n, c)).reshape(t, n, c)

    def forward(self, x):
        x = self.input_norm(x)
        x = x.permute(2, 0, 1)
        x = self.spike_encoder(x)
        x = self.proj_in(x)
        x = self._apply_time_bn(x, self.norm_in)
        x = self.drop_in(x)
        x = self.lif_in(x)

        for proj, norm, drop, lif in zip(self.projs, self.norms, self.drops, self.lifs):
            x = proj(x)
            x = self._apply_time_bn(x, norm)
            x = drop(x)
            x = lif(x)

        x = self.Classify(x)
        x = self.bn_readout(x.permute(1, 2, 0)).permute(2, 0, 1)
        x = x * self.readout_input_scale
        x = self.readout_neuron(x)
        x = x.mean(0)
        return x


class CUPY_SNN_LIF_PLIF_LIF_READOUT(nn.Module):
    def __init__(self, in_channels=22, out_num=4, beta=2, w=0.5, surrogate_function=surrogate.Sigmoid(),
                 time_step=128 * 3, readout_adapt_scale=0.02, readout_tau_adp_scale=6.0,
                 readout_v_threshold=0.2, readout_input_scale=2.5, lif_v_threshold=0.5,
                 lif_input_scale=2.5):
        super(CUPY_SNN_LIF_PLIF_LIF_READOUT, self).__init__()
        tau_mem = math.exp(-w) + 1
        channels = int(beta * in_channels)
        kernal = time_step // 32
        self.encode_C = nn.Conv1d(in_channels, channels, kernel_size=(1,), bias=False)
        self.encode_T = nn.Conv1d(channels, channels, kernel_size=(kernal,), padding=(kernal // 2,),
                                  groups=channels, bias=False)
        self.bn_T = nn.BatchNorm1d(channels)
        self.lif_input_scale = lif_input_scale
        self.lif = CUPYLIFNode(
            surrogate_function=surrogate_function,
            v_threshold=lif_v_threshold,
            v_reset=0.0,
            detach_reset=True,
            backend='torch',
            w=w,
        )
        self.plif = CUPYPLIFNode(init_tau=tau_mem, surrogate_function=surrogate_function)
        self.Classify = nn.Linear(in_features=channels, out_features=out_num)
        self.bn_readout = nn.BatchNorm1d(out_num)
        self.readout_input_scale = readout_input_scale
        self.readout_neuron = CUPYLIFNode(
            surrogate_function=surrogate_function,
            v_threshold=readout_v_threshold,
            v_reset=0.0,
            detach_reset=True,
            backend='torch',
            w=w,
        )
        functional.set_step_mode(self, step_mode='m')

    def forward(self, x):
        x = self.encode_C(x)
        x = self.encode_T(x)
        x = self.bn_T(x).permute(2, 0, 1)
        x = x * self.lif_input_scale
        x = self.lif(x)
        x = self.plif(x)
        x = self.Classify(x)
        x = self.bn_readout(x.permute(1, 2, 0)).permute(2, 0, 1)
        x = x * self.readout_input_scale
        x = self.readout_neuron(x)
        x = x.mean(0)
        return x


class CUPY_SNN_LIF_READOUT_STREAMING(nn.Module):
    """Streaming variant of CUPY_SNN_LIF_READOUT.

    - `forward(x, return_seq=False)` behaves like the original model when
      `return_seq=False` (averaged over time). If `return_seq=True` it
      returns per-time-step readout activations with shape `(N, T, C_out)`.
    - `early_decision(x, window_size, step, threshold)` performs a simple
      sliding-window early decision: for each sample, it computes the per-step
      class probabilities and the windowed average of the max-class probability;
      returns earliest time index where the windowed max-prob exceeds `threshold`.
    """

    def __init__(self, in_channels=22, out_num=4, beta=2, w=0.5, surrogate_function=surrogate.Sigmoid(),
                 time_step=128 * 3, readout_adapt_scale=0.02, readout_tau_adp_scale=6.0,
                 readout_v_threshold=0.2, readout_input_scale=2.5):
        super(CUPY_SNN_LIF_READOUT_STREAMING, self).__init__()
        tau_mem = math.exp(-w) + 1
        channels = int(beta * in_channels)
        kernal = time_step // 32
        self.encode_C = nn.Conv1d(in_channels, channels, kernel_size=(1,), bias=False)
        self.encode_T = nn.Conv1d(channels, channels, kernel_size=(kernal,), padding=(kernal // 2,),
                                  groups=channels, bias=False)
        self.bn_T = nn.BatchNorm1d(channels)
        self.neuron = CUPYPLIFNode(init_tau=tau_mem, surrogate_function=surrogate_function)
        self.Classify = nn.Linear(in_features=channels, out_features=out_num)
        self.bn_readout = nn.BatchNorm1d(out_num)
        self.readout_input_scale = readout_input_scale
        self.readout_neuron = CUPYLIFNode(
            surrogate_function=surrogate_function,
            v_threshold=readout_v_threshold,
            v_reset=0.0,
            detach_reset=True,
            backend='torch',
            w=w,
        )
        functional.set_step_mode(self, step_mode='m')

    def _compute_seq(self, x):
        # returns tensor of shape (T, N, out_num)
        x = self.encode_C(x)
        x = self.encode_T(x)
        x = self.bn_T(x).permute(2, 0, 1)
        x = self.neuron(x)
        x = self.Classify(x)
        x = self.bn_readout(x.permute(1, 2, 0)).permute(2, 0, 1)
        x = x * self.readout_input_scale
        x = self.readout_neuron(x)
        return x

    def forward(self, x, return_seq=False):
        seq = self._compute_seq(x)  # (T, N, C_out)
        if return_seq:
            # return (N, T, C_out)
            return seq.permute(1, 0, 2)
        return seq.mean(0)

    @torch.no_grad()
    def early_decision(self, x, window_size=50, step=1, threshold=0.8):
        """Sliding-window early decision.

        Returns a list of tuples for the batch: (decided_class or None, time_index or None).
        time_index is the last time step index included in the deciding window (0-based).
        """
        seq = self.forward(x, return_seq=True)  # (N, T, C)
        probs = torch.softmax(seq, dim=-1)  # (N, T, C)
        max_probs, max_idx = probs.max(dim=-1)  # (N, T)

        N, T = max_probs.shape
        if window_size <= 0 or window_size > T:
            raise ValueError('window_size must be in 1..T')

        # compute moving average of the max_probs for each sample
        # pad left with zeros so windowed average aligns with end of window
        kernel = torch.ones(window_size, device=seq.device, dtype=seq.dtype)
        results = []
        for n in range(N):
            p = max_probs[n]  # (T,)
            # convolution for moving average
            conv = torch.nn.functional.conv1d(p.view(1, 1, -1), kernel.view(1, 1, -1), padding=0)
            # conv has length T - window_size + 1, index i corresponds to window covering [i, i+window_size-1]
            conv = conv.view(-1) / float(window_size)

            decided = False
            for i in range(0, conv.shape[0], step):
                if conv[i].item() >= threshold:
                    # find class at the end of window (i + window_size -1)
                    t_idx = i + window_size - 1
                    decided_class = int(max_idx[n, t_idx].item())
                    results.append((decided_class, int(t_idx)))
                    decided = True
                    break
            if not decided:
                results.append((None, None))

        return results


class CUPY_SNN_PLIF_DUAL_READOUT(nn.Module):
    def __init__(self, in_channels=22, out_num=4, beta=2, w=0.5, surrogate_function=surrogate.Sigmoid(),
                 time_step=128 * 3, readout_adapt_scale=0.02, readout_tau_adp_scale=6.0,
                 readout_v_threshold=0.2, readout_input_scale=2.5, init_direct_weight=0.9):
        super(CUPY_SNN_PLIF_DUAL_READOUT, self).__init__()
        tau_mem = math.exp(-w) + 1
        tau_adp = tau_mem * readout_tau_adp_scale
        channels = int(beta * in_channels)
        kernal = time_step // 32
        self.encode_C = nn.Conv1d(in_channels, channels, kernel_size=(1,), bias=False)
        self.encode_T = nn.Conv1d(channels, channels, kernel_size=(kernal,), padding=(kernal // 2,),
                                  groups=channels, bias=False)
        self.bn_T = nn.BatchNorm1d(channels)
        self.neuron = CUPYPLIFNode(init_tau=tau_mem, surrogate_function=surrogate_function)

        # Direct readout keeps the original strong PLIF path.
        self.classify_direct = nn.Linear(in_features=channels, out_features=out_num)

        # ALIF readout adds temporal decision dynamics.
        self.classify_alif = nn.Linear(in_features=channels, out_features=out_num)
        self.bn_readout = nn.BatchNorm1d(out_num)
        self.readout_input_scale = readout_input_scale
        self.readout_neuron = ALIFNode(
            tau_mem=tau_mem,
            tau_adp=tau_adp,
            v_threshold=readout_v_threshold,
            adapt_scale=readout_adapt_scale,
            surrogate_function=surrogate_function,
            detach_reset=True,
        )

        # Start near direct PLIF logits and let ALIF branch improve only if useful.
        init_direct_weight = max(1e-3, min(1.0 - 1e-3, init_direct_weight))
        init_logit = math.log(init_direct_weight / (1.0 - init_direct_weight))
        self.fusion_logit = nn.Parameter(torch.tensor(init_logit, dtype=torch.float32))
        functional.set_step_mode(self, step_mode='m')

    def forward(self, x):
        x = self.encode_C(x)
        x = self.encode_T(x)
        x = self.bn_T(x).permute(2, 0, 1)
        x = self.neuron(x)

        direct_logits = self.classify_direct(x.mean(0))

        # Detach backbone features for ALIF branch to avoid destabilizing direct PLIF path.
        alif_logits = self.classify_alif(x.detach())
        alif_logits = self.bn_readout(alif_logits.permute(1, 2, 0)).permute(2, 0, 1)
        alif_logits = alif_logits * self.readout_input_scale
        alif_logits = self.readout_neuron(alif_logits)
        alif_logits = alif_logits.mean(0)

        alpha = torch.sigmoid(self.fusion_logit)
        return alpha * direct_logits + (1.0 - alpha) * alif_logits


class CUPY_SNN_3PLIF_LN_ALIF_READOUT(nn.Module):
    def __init__(self, in_channels=22, out_num=4, beta=2, w=0.5, surrogate_function=surrogate.Sigmoid(),
                 time_step=128 * 3, readout_adapt_scale=0.02, readout_tau_adp_scale=6.0,
                 readout_v_threshold=0.2, readout_input_scale=2.5):
        super(CUPY_SNN_3PLIF_LN_ALIF_READOUT, self).__init__()
        tau_mem = math.exp(-w) + 1
        tau_adp = tau_mem * readout_tau_adp_scale
        channels = int(beta * in_channels)
        kernal = time_step // 32
        self.encode_C = nn.Conv1d(in_channels, channels, kernel_size=(1,), bias=False)
        self.encode_T = nn.Conv1d(channels, channels, kernel_size=(kernal,), padding=(kernal // 2,),
                                  groups=channels, bias=False)
        self.ln_T = nn.LayerNorm(channels)
        self.neuron1 = CUPYPLIFNode(init_tau=tau_mem, surrogate_function=surrogate_function)
        self.neuron2 = CUPYPLIFNode(init_tau=tau_mem, surrogate_function=surrogate_function)
        self.neuron3 = CUPYPLIFNode(init_tau=tau_mem, surrogate_function=surrogate_function)
        self.Classify = nn.Linear(in_features=channels, out_features=out_num)
        self.ln_readout = nn.LayerNorm(out_num)
        self.readout_input_scale = readout_input_scale
        self.readout_neuron = ALIFNode(
            tau_mem=tau_mem,
            tau_adp=tau_adp,
            v_threshold=readout_v_threshold,
            adapt_scale=readout_adapt_scale,
            surrogate_function=surrogate_function,
            detach_reset=True,
        )
        functional.set_step_mode(self, step_mode='m')

    def forward(self, x):
        x = self.encode_C(x)
        x = self.encode_T(x).permute(2, 0, 1)
        x = self.ln_T(x)
        x = self.neuron1(x)
        x = self.neuron2(x)
        x = self.neuron3(x)
        x = self.Classify(x)
        x = self.ln_readout(x)
        x = x * self.readout_input_scale
        x = self.readout_neuron(x)
        x = x.mean(0)
        return x
