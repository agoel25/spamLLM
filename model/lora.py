"""Attributions:
- Hu et al. "LoRA: Low-Rank Adaptation of Large Language Models" (2021) https://arxiv.org/abs/2106.09685
- HuggingFace PEFT library: https://github.com/huggingface/peft
"""

import torch
import torch.nn as nn
import math

class LoRALayer(nn.Module):
    """lora layer

    instead of updating W directly, we learn W + BA where:
    - B shape is out_features x r
    - A shape is r x in_featrues
    - r << in_features and r << out_features
    
    """

    def __init__(self, original_layer: nn.Linear, r: int = 8, alpha: int = 16):
        super().__init__()
        self.original_layer = original_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        
        # initialize low rank matrices
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        
        # then we initialize A with kaiming and B with zeros
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5)) # learned how to do using ChatGPT
        nn.init.zeros_(self.lora_B)
        
        # make sure original weights stay unchanged
        self.original_layer.weight.requires_grad = False
        if self.original_layer.bias is not None:
            self.original_layer.bias.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # original output + lora
        # original path (stays frozenn)
        original_out = self.original_layer(x)

        # lora path: x @ A^T @ B^T
        lora_out = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return original_out + lora_out


def apply_lora_to_model(model, r: int = 8, alpha: int = 16, target_modules: list = None):
    """apply lora to specified modules in the model
    Args:
    - model: base LLM to use lora on
    - r: rank
    - alpha: scaling factor
    - target_modules: which layres we want to apply lora on
    """
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]  # keep default, using more leads to overfit
        # target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    # areeze all parameters first
    for param in model.parameters():
        param.requires_grad = False
    
    # replace all target layers with updated lora versions
    for name, module in model.named_modules(): # learned how to get named_modules from ChatGPT
        for target in target_modules:
            if name.endswith(target) and isinstance(module, nn.Linear):
                # learned how to do from ChatGPT
                parent_name = ".".join(name.split(".")[:-1])
                parent = model.get_submodule(parent_name) if parent_name else model
                attr_name = name.split(".")[-1]
                
                # replace with lora layer
                lora_layer = LoRALayer(module, r=r, alpha=alpha)
                setattr(parent, attr_name, lora_layer)
    
    return model


def get_lora_parameters(model):
    """get only the trainable lora parameters"""
    params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            params.append(param)
    return params