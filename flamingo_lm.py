import functools
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from transformers.models.mistral.modeling_mistral import MistralDecoderLayer
from transformers.utils import logging

from .helpers import GatedCrossAttentionBlock
from .utils import getattr_recursive, setattr_recursive

logger = logging.get_logger(__name__)


class FlamingoLayer(nn.Module):
    """
    FlamingoLayer is a wrapper around the GatedCrossAttentionBlock and DecoderLayer.
    """

    def __init__(
        self, gated_cross_attn_layer, decoder_layer, gradient_checkpointing=False
    ):
        super().__init__()
        self.gated_cross_attn_layer = gated_cross_attn_layer
        self.decoder_layer = decoder_layer
        self.vis_x = None
        self.media_locations = None
        if self.gated_cross_attn_layer is not None:
            self.gated_cross_attn_layer._use_gradient_checkpointing = (
                gradient_checkpointing
            )
        self.decoder_layer._use_gradient_checkpointing = gradient_checkpointing
        self._use_gradient_checkpointing = gradient_checkpointing
        if self._use_gradient_checkpointing:
            self.gradient_checkpointing_enable()

    def is_conditioned(self) -> bool:
        """Check whether the layer is conditioned."""
        return self.vis_x is not None and self.media_locations is not None

    # Used this great idea from this implementation of Flamingo (https://github.com/dhansmair/flamingo-mini/)
    def condition_vis_x(self, vis_x):
        self.vis_x = vis_x

    def condition_media_locations(self, media_locations):
        self.media_locations = media_locations

    def condition_use_cached_media(self, use_cached_media):
        self.use_cached_media = use_cached_media

    def forward(
        self,
        lang_x,
        attention_mask=None,
        **decoder_layer_kwargs,
    ):
        # Cross attention
        if self.gated_cross_attn_layer is not None:
            if self.vis_x is None:
                raise ValueError("vis_x must be conditioned before forward pass")

            if self.media_locations is None:
                raise ValueError(
                    "media_locations must be conditioned before forward pass"
                )

            lang_x = self.gated_cross_attn_layer(
                lang_x,
                self.vis_x,
                media_locations=self.media_locations,
                use_cached_media=self.use_cached_media,
            )

        # Normal decoder layer
        if (
            self._use_gradient_checkpointing
            and self.training
            and isinstance(self.decoder_layer, MistralDecoderLayer)
        ):
            if (
                "use_cache" in decoder_layer_kwargs
                and decoder_layer_kwargs["use_cache"] is True
            ):
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing."
                    " Setting `use_cache=False`..."
                )
                decoder_layer_kwargs["use_cache"] = False
            # lang_x = self._gradient_checkpointing_func(
            #     self.decoder_layer.__call__,
            #     lang_x, attention_mask=attention_mask, **decoder_layer_kwargs
            # )

            # Only work for Mistral
            lang_x = self._gradient_checkpointing_func(
                self.decoder_layer.__call__,
                lang_x,
                attention_mask,
                decoder_layer_kwargs["position_ids"],
                decoder_layer_kwargs["past_key_value"],
                decoder_layer_kwargs["output_attentions"],
                decoder_layer_kwargs["use_cache"],
            )
        else:
            lang_x = self.decoder_layer(
                lang_x, attention_mask=attention_mask, **decoder_layer_kwargs
            )
        return lang_x

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """
        Activates gradient checkpointing for the current model.

        Note that in other frameworks this feature can be referred to as "activation checkpointing" or "checkpoint
        activations".

        We pass the `__call__` method of the modules instead of `forward` because `__call__` attaches all the hooks of
        the module. https://discuss.pytorch.org/t/any-different-between-model-input-and-model-forward-input/3690/2

        Args:
            gradient_checkpointing_kwargs (dict, *optional*):
                Additional keyword arguments passed along to the `torch.utils.checkpoint.checkpoint` function.
        """
        if gradient_checkpointing_kwargs is None:
            gradient_checkpointing_kwargs = {}

        gradient_checkpointing_func = functools.partial(
            checkpoint, **gradient_checkpointing_kwargs
        )

        self._gradient_checkpointing_func = gradient_checkpointing_func

        if getattr(self, "_hf_peft_config_loaded", False):
            # When using PEFT + gradient checkpointing + Trainer we need to make sure the input has requires_grad=True
            # we do it also on PEFT: https://github.com/huggingface/peft/blob/85013987aa82aa1af3da1236b6902556ce3e483e/src/peft/peft_model.py#L334
            # When training with PEFT, only LoRA layers will have requires grad set to True, but the output of frozen layers need to propagate
            # the gradients to make sure the gradient flows.
            self.enable_input_require_grads()


class FlamingoLMMixin(nn.Module):
    """
    Mixin to add cross-attention layers to a language model.
    """

    def set_decoder_layers_attr_name(self, decoder_layers_attr_name):
        self.decoder_layers_attr_name = decoder_layers_attr_name

    def _get_decoder_layers(self):
        return getattr_recursive(self, self.decoder_layers_attr_name)

    def _set_decoder_layers(self, value):
        setattr_recursive(self, self.decoder_layers_attr_name, value)

    def init_flamingo(
        self,
        media_token_id,
        lang_hidden_size,
        vis_hidden_size,
        cross_attn_every_n_layers,
        *,
        enable_init_network_params=False,
        initializer_range=0.02,
        gradient_checkpointing=False,
    ):
        """
        Initialize Flamingo by adding a new gated cross attn to the decoder. Store the media token id for computing the media locations.
        """
        self.old_decoder_blocks = self._get_decoder_layers()
        self.gated_cross_attn_layers = nn.ModuleList(
            [
                (
                    GatedCrossAttentionBlock(
                        dim=lang_hidden_size,
                        dim_visual=vis_hidden_size,
                        ff_mult=4,
                        enable_init_network_params=enable_init_network_params,
                        initializer_range=initializer_range,
                        gradient_checkpointing=gradient_checkpointing,
                    )
                    if (layer_idx + 1) % cross_attn_every_n_layers == 0
                    else None
                )
                for layer_idx, _ in enumerate(self._get_decoder_layers())
            ]
        )
        self.init_flamingo_layers(gradient_checkpointing)
        self.media_token_id = media_token_id
        self.initialized_flamingo = True
        self._use_cached_vision_x = False
        self.gradient_checkpointing = gradient_checkpointing

    def init_flamingo_layers(self, gradient_checkpointing):
        """
        Re initializes the FlamingoLayers.
        Propagates any changes made to self.gated_corss_attn_layers or self.old_decoder_blocks
        """
        self._set_decoder_layers(
            nn.ModuleList(
                [
                    FlamingoLayer(
                        gated_cross_attn_layer, decoder_layer, gradient_checkpointing
                    )
                    for gated_cross_attn_layer, decoder_layer in zip(
                        self.gated_cross_attn_layers, self.old_decoder_blocks
                    )
                ]
            )
        )

    def forward(self, input_ids, attention_mask, **kwargs):
        """Condition the Flamingo layers on the media locations before forward()"""
        if not self.initialized_flamingo:
            raise ValueError(
                "Flamingo layers are not initialized. Please call `init_flamingo`"
                " first."
            )

        media_locations = input_ids == self.media_token_id

        # if there are media already cached and we're generating and there are no media tokens in the input,
        # we'll assume that ALL input tokens should attend to the last previous media that is cached.
        # this is especially important for HF generate() compatibility, since generate() calls forward()
        # repeatedly one token at a time (with no media tokens).
        # without this check, the model would not attend to any images when generating (after the first token)
        use_cached_media_locations = (
            self._use_cached_vision_x
            and self.is_conditioned()
            and not media_locations.any()
        )

        for layer in self._get_decoder_layers():
            if not use_cached_media_locations:
                layer.condition_media_locations(media_locations)
            layer.condition_use_cached_media(use_cached_media_locations)

        # package arguments for the other parent's forward. since we don't know the order of the arguments,
        # make them all kwargs
        kwargs["input_ids"] = input_ids
        kwargs["attention_mask"] = attention_mask

        # Mistral also need to set 'use_cache' to False when enable gradient checkpointing
        if self.gradient_checkpointing and isinstance(
            self.old_decoder_blocks[0], MistralDecoderLayer
        ):
            kwargs["use_cache"] = False
        return super().forward(**kwargs)  # Call the other parent's forward method

    def is_conditioned(self) -> bool:
        """Check whether all decoder layers are already conditioned."""
        return all(l.is_conditioned() for l in self._get_decoder_layers())

    def clear_conditioned_layers(self):
        for layer in self._get_decoder_layers():
            layer.condition_vis_x(None)
            layer.condition_media_locations(None)
            layer.condition_use_cached_media(None)
