import torch
import torch.nn as nn
from torchscale.architecture.config import RetNetConfig
from torchscale.architecture.retnet import RetNetRelPos
from torchscale.component.multiscale_retention import MultiScaleRetention
from torchscale.component.gate_linear_unit import GLU


class RetentionTempEncoderBlock(nn.Module):
    
    def __init__(self,
        hidden_dim:int = 128,
        num_head:int = 8,
        dropout:float = 0.,
    ):
        super().__init__()
        
        ret_args = RetNetConfig(
            decoder_embed_dim=hidden_dim,
            decoder_value_embed_dim=hidden_dim,
            decoder_retention_heads=num_head,
            decoder_ffn_embed_dim=hidden_dim*4,
            dropout=dropout,
            layernorm_eps=1e-8,
        )
        self.ret_args = ret_args
        self.ret_relpos = RetNetRelPos(ret_args)
        self.ret_unit = MultiScaleRetention(
            ret_args,
            ret_args.decoder_embed_dim,
            ret_args.decoder_value_embed_dim,
            ret_args.decoder_retention_heads,
        )
        self.ffn = GLU(
            embed_dim=hidden_dim,
            ffn_dim=ret_args.decoder_ffn_embed_dim,
            activation_fn=ret_args.activation_fn,
            dropout=dropout,
            activation_dropout=ret_args.activation_dropout
        )
        self.pre_norm1 = nn.LayerNorm(hidden_dim)
        self.pre_norm2 = nn.LayerNorm(hidden_dim)

    def forward(self,
        x: torch.Tensor,
        t: int = None,
        states: dict = None,
    ):
        if t is not None:
            # sequential
            if states is None:
                states = dict()
            residual = x
            x = self.pre_norm1(x)
            bsz, seq_len, d_model = x.shape
            assert seq_len == 1
            x = self.ret_unit.forward(
                x = x,
                rel_pos=self.ret_relpos(t, activate_recurrent=True),
                incremental_state=states
            )
            x = x + residual
            x = self.pre_norm2(x)
            x = self.ffn(x)
            x = x + residual
            return x
        else:
            # parallel
            residual = x
            x = self.pre_norm1(x)
            bsz, seq_len, d_model = x.shape
            x = self.ret_unit.forward(
                x = x,
                rel_pos=self.ret_relpos(seq_len),
            )
            x = x + residual
            x = self.pre_norm2(x)
            x = self.ffn(x)
            x = x + residual
            return x
