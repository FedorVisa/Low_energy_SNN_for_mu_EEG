"""Norse-based latency-coded SNN baseline models."""

import math

import torch
import torch.nn as nn
from norse.torch.module.lif import LIF, LIFParameters


class LatencySpikeEncoder(nn.Module):
    """Top-k latency encoder for EEG segments.

    Each channel is normalized inside a segment. The largest amplitudes are
    mapped to earlier spike times, producing one input spike train per EEG
    channel.
    """

    def __init__(self, time_step=250, topk=16, window_size=10, use_abs=True, eps=1e-6):
        super(LatencySpikeEncoder, self).__init__()
        self.time_step = int(time_step)
        self.topk = int(topk)
        self.window_size = int(window_size)
        self.use_abs = bool(use_abs)
        self.eps = float(eps)

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError("Expected input with shape (batch, channels, time)")

        batch, channels, time_len = x.shape
        values = x.abs() if self.use_abs else x
        min_v = values.amin(dim=-1, keepdim=True)
        max_v = values.amax(dim=-1, keepdim=True)
        norm = (values - min_v) / (max_v - min_v + self.eps)

        if self.window_size > 1:
            window = min(self.window_size, time_len)
            usable_len = (time_len // window) * window
            values = values[:, :, :usable_len]
            num_windows = usable_len // window
            windowed = values.reshape(batch, channels, num_windows, window)
            max_values = windowed.amax(dim=-1)
            min_values = windowed.amin(dim=-1)
            norm = (max_values - min_v) / (max_v - min_v + self.eps)
            norm = norm.clamp_(0.0, 1.0)
            local_latency = torch.round((1.0 - norm) * (window - 1)).long()
            window_offsets = torch.arange(
                num_windows,
                device=x.device,
                dtype=torch.long,
            ).view(1, 1, num_windows) * window
            latency = (window_offsets + local_latency).clamp_(0, self.time_step - 1)
            spikes = x.new_zeros((batch, channels, self.time_step))
            spikes.scatter_add_(2, latency, torch.ones_like(max_values))
            return spikes.clamp_(0.0, 1.0)

        k = max(1, min(self.topk, time_len))
        top_values = torch.topk(norm, k=k, dim=-1).values
        latency = torch.round((1.0 - top_values) * (self.time_step - 1)).long()
        latency = latency.clamp_(0, self.time_step - 1)

        spikes = x.new_zeros((batch, channels, self.time_step))
        spikes.scatter_add_(2, latency, torch.ones_like(top_values))
        return spikes.clamp_(0.0, 1.0)


class NORSE_LATENCY_CONV_LIF_READOUT(nn.Module):
    """Norse implementation of the requested latency-coded SNN.

    Architecture:
    input channel spike trains -> spatial 1x1 Conv1d -> temporal depthwise
    Conv1d -> Norse LIF hidden layer -> linear projection -> Norse LIF readout.
    The returned logits are output spike counts over the segment window.
    """

    def __init__(
        self,
        in_channels=22,
        out_num=4,
        beta=2,
        time_step=250,
        latency_topk=16,
        latency_window=10,
        lif_v_threshold=0.5,
        readout_v_threshold=0.2,
        lif_input_scale=2.5,
        readout_input_scale=2.5,
        tau_mem_ms=20.0,
        tau_syn_ms=5.0,
        dropout=0.0,
    ):
        super(NORSE_LATENCY_CONV_LIF_READOUT, self).__init__()
        channels = int(beta * in_channels)
        kernel = max(3, time_step // 32)
        if kernel % 2 == 0:
            kernel += 1

        self.encoder = LatencySpikeEncoder(
            time_step=time_step,
            topk=latency_topk,
            window_size=latency_window,
        )
        self.encode_C = nn.Conv1d(in_channels, channels, kernel_size=1, bias=False)
        self.encode_T = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel,
            padding=kernel // 2,
            groups=channels,
            bias=False,
        )
        self.bn_T = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(p=dropout) if dropout and dropout > 0 else nn.Identity()

        lif_params = LIFParameters(
            tau_mem_inv=torch.as_tensor(1000.0 / tau_mem_ms),
            tau_syn_inv=torch.as_tensor(1000.0 / tau_syn_ms),
            v_th=torch.as_tensor(float(lif_v_threshold)),
            v_reset=torch.as_tensor(0.0),
            method="super",
            alpha=torch.as_tensor(100.0),
        )
        readout_params = LIFParameters(
            tau_mem_inv=torch.as_tensor(1000.0 / tau_mem_ms),
            tau_syn_inv=torch.as_tensor(1000.0 / tau_syn_ms),
            v_th=torch.as_tensor(float(readout_v_threshold)),
            v_reset=torch.as_tensor(0.0),
            method="super",
            alpha=torch.as_tensor(100.0),
        )
        self.hidden_lif = LIF(lif_params)
        self.Classify = nn.Linear(channels, out_num)
        self.bn_readout = nn.BatchNorm1d(out_num)
        self.readout_lif = LIF(readout_params)
        self.lif_input_scale = float(lif_input_scale)
        self.readout_input_scale = float(readout_input_scale)
        self.time_step = int(time_step)

    def forward(self, x):
        x = self.encoder(x)
        x = self.encode_C(x)
        x = self.encode_T(x)
        x = self.bn_T(x)
        x = self.dropout(x).permute(2, 0, 1)

        hidden_seq, _ = self.hidden_lif(x * self.lif_input_scale)
        time_len, batch_size, _ = hidden_seq.shape
        logits = self.Classify(hidden_seq.reshape(time_len * batch_size, -1))
        logits = self.bn_readout(logits)
        logits = logits.reshape(time_len, batch_size, -1)
        readout_seq, _ = self.readout_lif(logits * self.readout_input_scale)
        spike_counts = readout_seq.sum(dim=0)
        return spike_counts / math.sqrt(float(self.time_step))
