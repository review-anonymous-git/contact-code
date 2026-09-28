"""Dual-channel predictor with separate timing and affective branches."""
import json
import math
from pathlib import Path

import torch
from torch import nn
from transformers import Qwen2Config, Qwen2ForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding

from .checkpoint import load_state


def time_encoding(times, dim):
    frequency = torch.exp(torch.arange(0, dim, 2, device=times.device).float()
                          * (-math.log(10000) / dim))
    phase = times.float().unsqueeze(-1) * frequency
    return torch.stack([phase.sin(), phase.cos()], -1).flatten(-2)[..., :dim]


class CausalSequence(nn.Module):
    def __init__(self, dim, heads, layers, dropout):
        super().__init__()
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.net = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)

    def forward(self, x, valid):
        x = x * valid.unsqueeze(-1)
        safe = valid.clone().bool()
        safe[:, 0] = True  # Avoid empty attention rows for padded examples.
        mask = torch.ones(x.shape[1], x.shape[1], device=x.device, dtype=torch.bool).triu(1)
        return self.net(x, mask=mask, src_key_padding_mask=~safe) * valid.unsqueeze(-1)


class MimiProjection(nn.Module):
    def __init__(self, feature_dim, hidden_dim):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(2 * feature_dim, hidden_dim), nn.GELU(),
                                  nn.Linear(hidden_dim, hidden_dim))

    def forward(self, channel0, channel1):
        return self.proj(torch.cat([channel0, channel1], -1))


class DualTurnCore(nn.Module):
    def __init__(self, config):
        super().__init__()
        qwen = Qwen2Config(**config["qwen"])
        qwen._attn_implementation = "sdpa"
        dim = qwen.hidden_size
        self.mimi_projection = MimiProjection(config["mimi_dim"], dim)
        self.backbone = Qwen2ForCausalLM(qwen)
        # Retain native heads so the checkpoint loads without discarded tensors.
        self.fvad_head = nn.Linear(dim, 8)
        for channel in (0, 1):
            setattr(self, f"vad_head_ch{channel}", nn.Linear(dim, 1))
            for task in ("eot", "bot", "hold", "bc"):
                setattr(self, f"{task}_head_ch{channel}", nn.Sequential(
                    nn.Linear(dim, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 1)))


class SharedBackbone(nn.Module):
    TASKS = ("vad", "fvad", "eot", "bot", "hold", "bc")

    def __init__(self, config):
        super().__init__()
        self.model = DualTurnCore(config)
        dim = config["qwen"]["hidden_size"]
        layers = config["qwen"]["num_hidden_layers"] + 1
        self.task_layer_weights = nn.ParameterDict({task: nn.Parameter(torch.zeros(layers))
                                                   for task in self.TASKS})
        self.fvad_head_256 = nn.Linear(dim, 256)

    def _fvad_head(self):
        return self.fvad_head_256

    def _hidden_from_features(self, channel0, channel1, tasks=None):
        embeddings = self.model.mimi_projection(channel0, channel1)
        # The language vocabulary head is not needed for audio prediction.
        result = self.model.backbone.model(inputs_embeds=embeddings, use_cache=False,
                                          output_hidden_states=True, return_dict=True)
        hidden = torch.stack(result.hidden_states, 0)
        return {task: (hidden * self.task_layer_weights[task].softmax(0)[:, None, None, None]).sum(0)
                for task in (tasks or self.TASKS)}


class SilenceHead(nn.Module):
    def __init__(self, dim, hidden, bins):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden),
                                 nn.GELU(), nn.Linear(hidden, bins))

    def forward(self, x):
        return self.net(x.float())


class ContactModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.base = SharedBackbone(config)
        source, dim = config["qwen"]["hidden_size"], config["hidden_dim"]
        args = (dim, config["attention_heads"])
        self.timing_in = nn.Linear(source, dim)
        self.timing = CausalSequence(*args, config["timing_layers"], config["dropout"])
        self.timing_out = nn.Linear(dim, source)
        self.silence_head = SilenceHead(dim, config["silence_hidden_dim"], config["silence_bins"])
        self.affect_in = nn.Linear(source, dim)
        self.affect = CausalSequence(*args, config["affect_layers"], config["dropout"])
        self.affect_head = nn.Linear(dim, 2 * 2 * 8)
        self.hidden_dim = dim

    def forward(self, channel0, channel1, available, valid):
        shared = self.base._hidden_from_features(channel0, channel1, ("fvad",))["fvad"]
        position = time_encoding(available, self.hidden_dim).to(shared)
        timing = self.timing(self.timing_in(shared) + position, valid)
        activity = self.base.fvad_head_256(shared + self.timing_out(timing)).float()
        affect = self.affect(self.affect_in(shared) + position, valid)
        factors = self.affect_head(affect).float().reshape(*affect.shape[:2], 2, 2, 8)
        joint = (factors[..., 0, :, None] + factors[..., 1, None, :]).reshape(
            *affect.shape[:2], 2, 64)
        return dict(activity=activity, silence=self.silence_head(timing),
                    affect=joint, affect_factors=factors)


def load_model(checkpoint, config_path, device="cpu"):
    config = json.loads(Path(config_path).read_text())
    with torch.device("meta"):
        model = ContactModel(config)
    model.load_state_dict(load_state(checkpoint), strict=True, assign=True)
    backbone = model.base.model.backbone
    backbone.model.rotary_emb = Qwen2RotaryEmbedding(backbone.config, device="cpu")
    backbone.tie_weights()
    return model.to(device).eval().requires_grad_(False)
