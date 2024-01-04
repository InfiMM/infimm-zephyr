# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
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
"""
Processor class for InfiMMZephyr.
"""

import random
from typing import List, Optional, Tuple, Union
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torchvision.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    Normalize,
    Resize,
    ToTensor,
)

from transformers import AutoTokenizer
from transformers.image_processing_utils import ImageProcessingMixin
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import BatchEncoding

IMAGE_TOKEN = "<image>"
END_OF_CHUNK_TOKEN = "<|endofchunk|>"

OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)


def _convert_to_rgb(image):
    return image.convert("RGB")


class ResizeKeepRatio:
    """Resize and Keep Ratio

    Copy & paste from `timm`
    """

    def __init__(
        self,
        size,
        longest=0.0,
        interpolation=InterpolationMode.BICUBIC,
        random_scale_prob=0.0,
        random_scale_range=(0.85, 1.05),
        random_aspect_prob=0.0,
        random_aspect_range=(0.9, 1.11),
    ):
        if isinstance(size, (list, tuple)):
            self.size = tuple(size)
        else:
            self.size = (size, size)
        self.interpolation = interpolation
        self.longest = float(longest)  # [0, 1] where 0 == shortest edge, 1 == longest
        self.random_scale_prob = random_scale_prob
        self.random_scale_range = random_scale_range
        self.random_aspect_prob = random_aspect_prob
        self.random_aspect_range = random_aspect_range

    @staticmethod
    def get_params(
        img,
        target_size,
        longest,
        random_scale_prob=0.0,
        random_scale_range=(0.85, 1.05),
        random_aspect_prob=0.0,
        random_aspect_range=(0.9, 1.11),
    ):
        """Get parameters"""
        source_size = img.size[::-1]  # h, w
        h, w = source_size
        target_h, target_w = target_size
        ratio_h = h / target_h
        ratio_w = w / target_w
        ratio = max(ratio_h, ratio_w) * longest + min(ratio_h, ratio_w) * (
            1.0 - longest
        )
        if random_scale_prob > 0 and random.random() < random_scale_prob:
            ratio_factor = random.uniform(random_scale_range[0], random_scale_range[1])
            ratio_factor = (ratio_factor, ratio_factor)
        else:
            ratio_factor = (1.0, 1.0)
        if random_aspect_prob > 0 and random.random() < random_aspect_prob:
            aspect_factor = random.uniform(
                random_aspect_range[0], random_aspect_range[1]
            )
            ratio_factor = (
                ratio_factor[0] / aspect_factor,
                ratio_factor[1] * aspect_factor,
            )
        size = [round(x * f / ratio) for x, f in zip(source_size, ratio_factor)]
        return size

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Resized, padded to at least target size, possibly cropped to exactly target size
        """
        size = self.get_params(
            img,
            self.size,
            self.longest,
            self.random_scale_prob,
            self.random_scale_range,
            self.random_aspect_prob,
            self.random_aspect_range,
        )
        img = F.resize(img, size, self.interpolation)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + "(size={0}".format(self.size)
        format_string += f", interpolation={self.interpolation})"
        format_string += f", longest={self.longest:.3f})"
        return format_string


def image_transform(
    image_size: Union[int, Tuple[int, int]],
    mean: Optional[Tuple[float, ...]] = None,
    std: Optional[Tuple[float, ...]] = None,
    resize_mode: Optional[str] = None,
    interpolation: Optional[str] = None,
):
    mean = mean or OPENAI_DATASET_MEAN
    if not isinstance(mean, (list, tuple)):
        mean = (mean,) * 3

    std = std or OPENAI_DATASET_STD
    if not isinstance(std, (list, tuple)):
        std = (std,) * 3

    interpolation = interpolation or "bicubic"
    assert interpolation in ["bicubic", "bilinear", "random"]
    # NOTE random is ignored for interpolation_mode, so defaults to BICUBIC for inference if set
    interpolation_mode = (
        InterpolationMode.BILINEAR
        if interpolation == "bilinear"
        else InterpolationMode.BICUBIC
    )

    resize_mode = resize_mode or "shortest"
    assert resize_mode in ("shortest", "longest", "squash")

    normalize = Normalize(mean=mean, std=std)

    assert resize_mode == "shortest"
    if not isinstance(image_size, (tuple, list)):
        image_size = (image_size, image_size)
    if image_size[0] == image_size[1]:
        # simple case, use torchvision built-in Resize w/ shortest edge mode (scalar size arg)
        transforms = [Resize(image_size[0], interpolation=interpolation_mode)]
    else:
        # resize shortest edge to matching target dim for non-square target
        transforms = [ResizeKeepRatio(image_size)]
    transforms += [CenterCrop(image_size)]

    transforms.extend(
        [
            _convert_to_rgb,
            ToTensor(),
            normalize,
        ]
    )
    return Compose(transforms)


class EVAClipImageProcessor(ImageProcessingMixin):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.processor = image_transform(image_size=336)

    def _prepare_images(self, batch: List[List[Image.Image]]) -> torch.Tensor:
        """
        Convert images to tensors, reshape them, and stack them.
        Args:
            batch: A list of lists of images.
        Returns:
            preprocessed images (tensors) or None
                shape (B, T_img, F, C, H, W)
                None if no images in batch
        """
        images_per_example = max(len(x) for x in batch)
        batch_images = None
        for iexample, example in enumerate(batch):
            for iimage, image in enumerate(example):
                preprocessed = self.processor(image)
                if batch_images is None:
                    batch_images = torch.zeros(
                        (len(batch), images_per_example, 1) + preprocessed.shape,
                        dtype=preprocessed.dtype,
                    )
                batch_images[iexample, iimage, 0] = preprocessed
        return batch_images

    def preprocess(self, imgpaths=None):
        if imgpaths is None or len(imgpaths) == 0:
            images = [(Image.new("RGB", (336, 336), color="black"))]
        else:
            images = [Image.open(fp) for fp in imgpaths]
        return self._prepare_images([images])


class InfiMMZephyrProcessor(ProcessorMixin):
    r"""
    Constructs a InfiMMZephyr processor which wraps a tokenizer and an image processor into a single processor.

    Args:
        image_processor (`EVAClipImageProcessor`):
            An instance of [`EVAClipImageProcessor`]. The image processor is a required input.
        tokenizer (`LlamaTokenizer`):
            An instance of [`LlamaTokenizer`]. The tokenizer is a required input.
        image_size (`int`, *optional*, defaults to 336): Image size (assuming a square image)
    """

    attributes = ["tokenizer"]
    tokenizer_class = "LlamaTokenizer"

    def __init__(self, tokenizer=None, **kwargs):
        self.image_processor = EVAClipImageProcessor()
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained("infimm-zephyr", verbose=False)

        super().__init__(tokenizer, tokenizer)

    def _prepare_text(
        self,
        batch: List[List[str]],
        padding="longest",
        truncation=True,
        max_length=2048,
    ):
        """
        Tokenize the text and stack them.
        Args:
            batch: A list of lists of strings.
        Returns:
            input_ids (tensor)
                shape (B, T_txt)
            attention_mask (tensor)
                shape (B, T_txt)
        """
        encodings = self.tokenizer(
            batch,
            padding=padding,
            truncation=truncation,
            return_tensors="pt",
            max_length=max_length,
        )
        input_ids, attention_mask = encodings["input_ids"], encodings["attention_mask"]
        return input_ids, attention_mask

    def __call__(
        self,
        prompts,
    ) -> BatchEncoding:
        """This method takes batched or non-batched prompts made of text and images and converts them into prompts that
        the model was trained on and prepares the image pixel values for the model to process.
        """
        image_paths = self._extract_image_paths(prompts)
        images = self.image_processor.preprocess(image_paths)
        prompts = self._replace_with_media_tokens(prompts)
        final_prompt = self.apply_chat_template(prompts)
        input_ids, attention_mask = self._prepare_text([final_prompt])
        return BatchEncoding(
            data={
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "batch_images": images,
            }
        )

    def _extract_image_paths(self, prompts):
        image_paths = []
        for round in prompts:
            if round["role"] != "user":
                continue
            for piece in round["content"]:
                if isinstance(piece, dict):
                    image_paths.append(piece["image"])
        return image_paths

    def _replace_with_media_tokens(self, prompts):
        new_prompts = []
        for round in prompts:
            if round["role"] != "user":
                new_prompts.append(round)
            new_content = []
            for piece in round["content"]:
                if isinstance(piece, dict):
                    new_content.append(f"{END_OF_CHUNK_TOKEN}{IMAGE_TOKEN}")
                else:
                    new_content.append(piece)
            new_prompts.append({"role": "user", "content": "".join(new_content)})
        return new_prompts

    def apply_chat_template(self, messages, task="generation"):
        if messages[0]["role"] != "system":
            messages.insert(0, {"role": "system", "content": ""})
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True if task == "generation" else False,
        )
        return prompt

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))
