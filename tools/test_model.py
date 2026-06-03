"""Small standalone model definition used for local architecture checks."""

import torch
import torch.nn as nn
from norse.torch.module.lif import LIFCell, LIFParameters
from model.my_lif import FixedPointQuantizer, GoldenLIFNeuronFixedPoint


class SNNVoiceCommand(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 3,
        norm: str = "layer",  # 'layer' | 'batch' | 'none'
        dropout: float = 0.0,
        lif_parameters: LIFParameters | None = None,
        readout_alpha: float = 1.0,
        use_bias: bool = True,
    ):
        """
        Feed-forward SNN with LIF cells. Linear layers provide synaptic weights,
        LIF cells provide spiking dynamics. Outputs are time-averaged logits.

        Args:
            input_size: feature dimension per time step (e.g., n_mels)
            hidden_size: number of LIF neurons per hidden layer
            output_size: number of classes
            num_layers: total layers including input+hidden; must be >= 2
        """
        super().__init__()

        if num_layers < 2:
            raise ValueError("num_layers must be >= 2")

        # Input projection + optional norm/dropout + LIF
        self.fc_in = nn.Linear(input_size, hidden_size, bias=use_bias)
        self.norm_kind = (norm or "none").lower()
        if self.norm_kind == "batch":
            self.norm_in = nn.BatchNorm1d(hidden_size)
        elif self.norm_kind == "layer":
            self.norm_in = nn.LayerNorm(hidden_size)
        else:
            self.norm_in = None
        self.drop_in = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        lif_parameters = lif_parameters or LIFParameters()
        self.lif_in = LIFCell(lif_parameters)

        # Hidden projections + optional norms/dropouts + LIFs
        hidden_count = max(0, num_layers - 2)
        self.fcs = nn.ModuleList([nn.Linear(hidden_size, hidden_size, bias=use_bias) for _ in range(hidden_count)])
        if self.norm_kind == "batch":
            self.norms = nn.ModuleList([nn.BatchNorm1d(hidden_size) for _ in range(hidden_count)])
        elif self.norm_kind == "layer":
            self.norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(hidden_count)])
        else:
            self.norms = nn.ModuleList([nn.Identity() for _ in range(hidden_count)])
        self.drops = nn.ModuleList([nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity() for _ in range(hidden_count)])
        self.lifs = nn.ModuleList([LIFCell(lif_parameters) for _ in range(hidden_count)])

        # Readout
        self.fc_out = nn.Linear(hidden_size, output_size, bias=use_bias)
        self.readout_alpha = float(readout_alpha)

    def _ensure_b_t_f(self, x: torch.Tensor, input_size: int) -> torch.Tensor:
        """Ensure tensor shape (B, T, F). Accepts (B, T, F), (B, F, T), or (F, T)."""
        if x.dim() == 2:  # (F, T) -> add batch dim and transpose to (B, T, F)
            f, t = x.shape
            if f != input_size:
                # assume (T, F)
                t, f = x.shape
                x = x.view(1, t, f)
            else:
                x = x.t().unsqueeze(0)
        elif x.dim() == 3:
            b, a, c = x.shape
            # If middle dim is input_size, we likely have (B, F, T) -> permute
            if a == input_size and c != input_size:
                x = x.permute(0, 2, 1)
            # else assume already (B, T, F)
        else:
            raise ValueError("Input tensor must be 2D or 3D (F,T) or (B,T,F)/(B,F,T)")
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, F)
        x = self._ensure_b_t_f(x, self.fc_in.in_features)
        b, t, f = x.shape

        # LIF states
        s_in = None
        hidden_states = [None for _ in self.lifs]

        logits_accum = []
        for step in range(t):
            cur = x[:, step, :]
            cur = self.fc_in(cur)
            if self.norm_in is not None:
                # BatchNorm1d expects (N, C); LayerNorm works on last dim
                cur = self.norm_in(cur)
            cur = self.drop_in(cur)
            spk, s_in = self.lif_in(cur, s_in)

            for i, (fc, lif) in enumerate(zip(self.fcs, self.lifs)):
                cur = fc(spk)
                cur = self.norms[i](cur)
                cur = self.drops[i](cur)
                spk, hidden_states[i] = lif(cur, hidden_states[i])

            # Backward-compatible readout: include membrane potential contribution.
            v_last = hidden_states[-1].v if hidden_states else s_in.v
            readout_vec = spk + self.readout_alpha * v_last
            logits = self.fc_out(readout_vec)
            logits_accum.append(logits)

        # Time-average logits
        out = torch.stack(logits_accum, dim=1).mean(dim=1)
        return out
