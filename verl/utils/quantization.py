# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple


import os

@dataclass
class PseudoQuantConfig:
    """Configuration for pseudo quantization during training."""
    enable: bool = False
    group_size: int = 128
    quant_method: str = "simple_4bit"  # only supports simple 4bit now


def is_pseudo_quant_enabled_from_env() -> bool:
    """Check if pseudo quantization is enabled via environment variable."""
    return os.getenv("PSEUDO_QUANT_ENABLE", "0") == "1"


def get_pseudo_quant_group_size() -> int:
    """Get group size from environment variable or use default."""
    return int(os.getenv("PSEUDO_QUANT_GROUP_SIZE", "128"))


def _quantize_4bit(weight: torch.Tensor, group_size: int = 128) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Simple 4bit per-block quantization.

    Args:
        weight: [out_dim, in_dim] - original weight in fp16/bf16
        group_size: block size for quantization

    Returns:
        quanted: int32 tensor where each 32bits packs 8 4bit values
        scales: per-block scaling factor
        zeros: per-block zero point
    """
    out_dim, in_dim = weight.shape
    assert in_dim % group_size == 0, f"input dim {in_dim} not divisible by group_size {group_size}"

    # Reshape to blocks: [out_dim, num_blocks, group_size]
    num_blocks = in_dim // group_size
    weight_reshaped = weight.reshape(out_dim, num_blocks, group_size)

    # Find min/max per block
    w_min = weight_reshaped.amin(dim=-1, keepdim=True)  # [out_dim, num_blocks, 1]
    w_max = weight_reshaped.amax(dim=-1, keepdim=True)

    # Calculate scale and zero point for 4bit (0-15 range)
    scales = (w_max - w_min) / 15.0  # [out_dim, num_blocks, 1]
    zeros = torch.round(-w_min / scales).clamp_(0, 15)  # [out_dim, num_blocks, 1]

    # Quantize
    quantized = torch.round((weight_reshaped - w_min) / scales + zeros).to(torch.int32)  # 0-15

    # Pack 8 4bit values into one int32
    # output shape: [out_dim, num_blocks, (group_size + pack_num - 1) // pack_num]
    pack_num = 32 // 4  # 8 values per int32
    quanted_out_shape = (out_dim, num_blocks, (group_size + pack_num - 1) // pack_num)
    quanted = torch.zeros(quanted_out_shape, dtype=torch.int32, device=weight.device)

    for pack_idx in range(pack_num):
        start = pack_idx * 4
        for block_col in range(group_size // pack_num):
            col_idx = block_col * pack_num + pack_idx
            quanted[:, :, block_col] |= (quantized[:, :, col_idx] << start)


    return quanted, scales.squeeze(-1), zeros.squeeze(-1)


def _dequantize_4bit(quanted: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor,
                      out_shape: Tuple[int, int], group_size: int = 128) -> torch.Tensor:
    """
    Dequantize from 4bit back to original dtype.

    Args:
        quanted: packed 4bit quantized weights
        scales: per-block scales
        zeros: per-block zero points
        out_shape: original shape [out_dim, in_dim]
        group_size: block size used in quantization

    Returns:
        dequanted: dequantized weight in original dtype
    """
    out_dim, in_dim = out_shape
    num_blocks = in_dim // group_size
    pack_num = 32 // 4  # 8 4bit per int32

    dequanted = torch.zeros(out_shape, dtype=scales.dtype, device=quanted.device)

    for i in range((group_size + pack_num - 1) // pack_num):
        shift = i * 4
        mask = (15 << shift)
        extracted = (quanted[:, :, i] & mask) >> shift  # [out_dim, num_blocks]
        for j in range(group_size // pack_num):
            block_idx = j * pack_num + i
            dequanted[:, block_idx * group_size:(block_idx + 1) * group_size] = (
                extracted - zeros) * scales

    return dequanted


class PseudoQuantizeSTE(torch.autograd.Function):
    """
    Pseudo quantization with Straight-Through Estimator gradient.

    Forward: quantize -> dequantize (add quantization error)
    Backward: straight through gradient (dL/dw_out = dL/dw_in)
    """
    @staticmethod
    def forward(ctx, weight: torch.Tensor, group_size: int):
        # Forward: quantize -> dequantize
        quanted, scales, zeros = _quantize_4bit(weight, group_size)
        dequanted = _dequantize_4bit(quanted, scales, zeros, weight.shape, group_size)
        # Save nothing for backward - gradient is straight through
        return dequanted.to(weight.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        # STE: gradient passes through unchanged
        return grad_output, None


def pseudo_quantize_weight(weight: torch.Tensor, group_size: int = 128) -> torch.Tensor:
    """Pseudo quantization with STE for training"""
    return PseudoQuantizeSTE.apply(weight, group_size)


def register_pseudo_quant_hooks(model: nn.Module, config: PseudoQuantConfig):
    """
    Register forward hook for MLP Linear layers to do pseudo quantization.
    Only applies to MLP/GroupMLP layers, skips attention linear layers.

    Each forward pass does:
    1. Save original fp16/bf16 weight
    2. quantize → dequantize to get pseudo-quantized weight (adds quantization error)
    3. Run forward with pseudo-quantized weight
    4. Restore original weight
    5. Gradient is straight-through estimated (STE)

    This simulates the quantization error that will occur at inference time,
    allowing the model to adapt to it during training.
    """
    if not config.enable:
        return

    group_size = config.group_size

    # Only quantize MLP/GroupMLP related linear layers
    mlp_keywords = ['mlp', 'fc', 'proj', 'feedforward', 'ffn', 'groupmlp']

    def _pseudo_quant_forward_hook(module: nn.Linear, input, output):
        # Save original weight
        orig_weight = module.weight.data

        # Pseudo quantization through STE
        dequanted = pseudo_quantize_weight(orig_weight, group_size)

        # Replace weight for this forward
        module.weight.data = dequanted

        # Recompute forward with pseudo-quantized weight
        # Hook runs after module forward, need to recompute
        output = module(input[0])

        # Restore original weight for next iteration and gradient update
        module.weight.data = orig_weight

        return output

    # Register hook only to MLP Linear layers
    num_hooks = 0
    for name, module in model.named_modules():
        name_lower = name.lower()
        if isinstance(module, nn.Linear) and any(k in name_lower for k in mlp_keywords):
            module.register_forward_hook(_pseudo_quant_forward_hook)
            num_hooks += 1

    print(f"[verl][quantization] Registered pseudo-quantization hooks for {num_hooks} MLP Linear layers")


def apply_online_int4_quantization(full_weights: dict, group_size: int = 128) -> dict:
    """
    Apply true 4bit quantization to full weights for sending to vLLM.
    Only quantizes MLP/GroupMLP linear weights, skip other layers.
    Quantized weights are split into multiple keys with suffix:
    - {name}_quanted: packed 4bit weights
    - {name}_scales: per-group scaling factors
    - {name}_zeros: per-group zero points
    - {name}_shape: original shape

    Args:
        full_weights: dictionary of full weights gathered from training engine
        group_size: block size for 4bit quantization

    Returns:
        dictionary: quantized MLP weights split to multiple keys,
                    unchanged for other layers
    """
    quantized_weights = {}
    # Only quantize MLP/GroupMLp related linear weights
    mlp_keywords = ['mlp', 'fc', 'proj', 'feedforward', 'ffn', 'groupmlp']

    for name, weight in full_weights.items():
        name_lower = name.lower()
        # Check if this is a MLP linear layer
        is_mlp_linear = (
            any(keyword in name_lower for keyword in mlp_keywords) and
            'weight' in name_lower and
            len(weight.shape) == 2  # Linear weight is [out, in]
        )

        if is_mlp_linear:
            out_dim, in_dim = weight.shape
            if in_dim % group_size != 0:
                # If not divisible, don't quantize
                quantized_weights[name] = weight
            else:
                quanted, scales, zeros = _quantize_4bit(weight, group_size)
                # Split into separate keys for easy loading
                quantized_weights[f"{name}_quanted"] = quanted
                quantized_weights[f"{name}_scales"] = scales
                quantized_weights[f"{name}_zeros"] = zeros
                quantized_weights[f"{name}_shape"] = torch.tensor(weight.shape, dtype=torch.int64)
        else:
            # Don't quantize: attention, layernorm, embeddings, lm_head, etc.
            quantized_weights[name] = weight

    return quantized_weights
