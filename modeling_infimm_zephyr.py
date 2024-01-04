import importlib
import math
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Generator, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.cuda.amp import autocast

from transformers import GenerationConfig, PreTrainedTokenizer, StoppingCriteriaList
from transformers.generation.logits_process import LogitsProcessorList

if TYPE_CHECKING:
    from transformers.generation.streamers import BaseStreamer

from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.models.mistral.configuration_mistral import MistralConfig
from transformers.models.mistral.modeling_mistral import MistralForCausalLM
from transformers.utils import logging

try:
    from einops import rearrange
except ImportError:
    rearrange = None
from torch import nn

from .configuration_infimm_zephyr import InfiMMConfig
from .eva_vit import CLIPVisionCfg, EVAVisionTransformer
from .flamingo import Flamingo
from .flamingo_lm import FlamingoLMMixin
from .helpers import PerceiverResampler
from .utils import _infer_decoder_layers_attr_name, extend_instance

SUPPORT_CUDA = torch.cuda.is_available()
SUPPORT_BF16 = SUPPORT_CUDA and torch.cuda.is_bf16_supported()
SUPPORT_FP16 = SUPPORT_CUDA and torch.cuda.get_device_capability(0)[0] >= 7


class InfiMMPreTrainedModel(PreTrainedModel):
    config_class = InfiMMConfig
    base_model_prefix = "transformer"
    is_parallelizable = False
    supports_gradient_checkpointing = True

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)


class InfiMMZephyrModel(InfiMMPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.vision_config = config.visual
        vision_encoder = self.build_vision_encoder()
        self.language_config = config.language
        language_encoder = self.build_language_encoder()

        self.model = self.build_flamingo(vision_encoder, language_encoder)

    def build_vision_encoder(self):
        vision_cfg = CLIPVisionCfg(**self.vision_config)

        vision_encoder = EVAVisionTransformer(
            img_size=vision_cfg.image_size,
            patch_size=vision_cfg.patch_size,
            num_classes=vision_cfg.embed_dim,
            use_mean_pooling=vision_cfg.global_average_pool,  # False
            init_values=vision_cfg.ls_init_value,
            patch_dropout=vision_cfg.patch_dropout,
            embed_dim=vision_cfg.width,
            depth=vision_cfg.layers,
            num_heads=vision_cfg.width // vision_cfg.head_width,
            mlp_ratio=vision_cfg.mlp_ratio,
            qkv_bias=vision_cfg.qkv_bias,
            drop_path_rate=vision_cfg.drop_path_rate,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            xattn=vision_cfg.xattn,
            rope=vision_cfg.rope,
            postnorm=vision_cfg.postnorm,
            pt_hw_seq_len=vision_cfg.pt_hw_seq_len,  # 224/14
            intp_freq=vision_cfg.intp_freq,
            naiveswiglu=vision_cfg.naiveswiglu,
            subln=vision_cfg.subln,
        )

        return vision_encoder

    def build_language_encoder(self):
        mistral_config = MistralConfig(**self.language_config)
        lang_encoder = MistralForCausalLM(mistral_config)
        return lang_encoder

    def build_flamingo(self, vision_encoder, lang_encoder):
        extend_instance(lang_encoder, FlamingoLMMixin)

        decoder_layers_attr_name = _infer_decoder_layers_attr_name(lang_encoder)
        lang_encoder.set_decoder_layers_attr_name(decoder_layers_attr_name)
        # lang_encoder.resize_token_embeddings(self.config.)

        model = Flamingo(
            vision_encoder,
            lang_encoder,
            self.config.eoc_token_id,
            self.config.image_token_id,
            vis_dim=self.vision_config["width"],
            cross_attn_every_n_layers=self.config.cross_attn_every_n_layers,
            gradient_checkpointing=self.config.use_grad_checkpoint,
        )

        return model

    def generate(
        self,
        input_ids,
        attention_mask,
        batch_images,
        min_generation_length: int,
        max_generation_length: int,
        **kwargs,
    ):
        with torch.inference_mode():
            outputs = self.model.generate(
                batch_images,
                input_ids,
                attention_mask,
                min_new_tokens=min_generation_length,
                max_new_tokens=max_generation_length,
                **kwargs,
            )

        # Extract only the new gnerated tokens
        outputs = outputs[:, len(input_ids[0]) :]
        return outputs
