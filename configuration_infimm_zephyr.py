# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from transformers import PretrainedConfig


class InfiMMConfig(PretrainedConfig):
    model_type = "infimm"

    def __init__(
        self,
        model_type="infimm-zephyr",
        seq_length=1024,
        tokenizer_type="ZephyrTokenizer",
        torch_dtype="bfloat16",
        transformers_version="4.35.2",
        use_cache=True,
        use_flash_attn=False,
        cross_attn_every_n_layers=2,
        use_grad_checkpoint=False,
        freeze_llm=True,
        visual=None,
        language=None,
        image_token_id=None,
        eoc_token_id=None,
        **kwargs,
    ):
        self.model_type = model_type
        self.seq_length = seq_length
        self.tokenizer_type = tokenizer_type
        self.torch_dtype = torch_dtype
        self.transformers_version = transformers_version
        self.use_cache = use_cache
        self.use_flash_attn = use_flash_attn
        self.cross_attn_every_n_layers = cross_attn_every_n_layers
        self.use_grad_checkpoint = use_grad_checkpoint
        self.freeze_llm = freeze_llm
        self.visual = visual
        self.language = language
        self.image_token_id = image_token_id
        self.eoc_token_id = eoc_token_id
        super().__init__(**kwargs)
