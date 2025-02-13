# The MIT License (MIT)

# Copyright (C) 2021-2023 ExplosionAI GmbH

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from typing import Any, Dict, List, Mapping, Optional, Tuple

from ...util.string import StringTransform, StringTransformations
from ..hf_hub.conversion import (
    CommonHFKeys,
    HFConfigKey,
    HFConfigKeyDefault,
    HFSpecificConfig,
    config_from_hf,
    config_to_hf,
)
from .config import BERTConfig

# Order-dependent.
HF_PARAM_KEY_TRANSFORMS: List[StringTransform] = [
    # Old HF parameter names (one-way transforms).
    StringTransformations.regex_sub((r"\.gamma$", ".weight"), backward=None),
    StringTransformations.regex_sub((r"\.beta$", ".bias"), backward=None),
    # Prefixes.
    StringTransformations.remove_prefix("bert.", reversible=False),
    StringTransformations.regex_sub(
        (r"^encoder\.(layer\.)", "\\1"),
        (r"^(layer\.)", "encoder.\\1"),
    ),
    # Layers.
    StringTransformations.regex_sub((r"^layer", "layers"), (r"^layers", "layer")),
    # Attention blocks.
    StringTransformations.regex_sub(
        (r"\.attention\.self\.(query|key|value)", ".mha.\\1"),
        (r"\.mha\.(query|key|value)", ".attention.self.\\1"),
    ),
    StringTransformations.sub(".attention.output.dense", ".mha.output"),
    StringTransformations.sub(
        r".attention.output.LayerNorm", ".attn_residual_layer_norm"
    ),
    # Pointwise feed-forward layers.
    StringTransformations.sub(".intermediate.dense", ".ffn.intermediate"),
    StringTransformations.regex_sub(
        (r"(\.\d+)\.output\.LayerNorm", "\\1.ffn_residual_layer_norm"),
        (r"(\.\d+)\.ffn_residual_layer_norm", "\\1.output.LayerNorm"),
    ),
    StringTransformations.regex_sub(
        (r"(\.\d+)\.output\.dense", "\\1.ffn.output"),
        (r"(\.\d+)\.ffn\.output", "\\1.output.dense"),
    ),
    # Embeddings.
    StringTransformations.replace(
        "embeddings.word_embeddings.weight", "embeddings.piece_embeddings.weight"
    ),
    StringTransformations.replace(
        "embeddings.token_type_embeddings.weight", "embeddings.type_embeddings.weight"
    ),
    StringTransformations.replace(
        "embeddings.position_embeddings.weight", "embeddings.position_embeddings.weight"
    ),
    StringTransformations.replace(
        "embeddings.LayerNorm.weight", "embeddings.embed_output_layer_norm.weight"
    ),
    StringTransformations.replace(
        "embeddings.LayerNorm.bias", "embeddings.embed_output_layer_norm.bias"
    ),
]

HF_CONFIG_KEYS: List[Tuple[HFConfigKey, Optional[HFConfigKeyDefault]]] = [
    (CommonHFKeys.ATTENTION_PROBS_DROPOUT_PROB, None),
    (CommonHFKeys.HIDDEN_DROPOUT_PROB, None),
    (CommonHFKeys.HIDDEN_SIZE, None),
    (CommonHFKeys.HIDDEN_ACT, None),
    (CommonHFKeys.INTERMEDIATE_SIZE, None),
    (CommonHFKeys.LAYER_NORM_EPS, None),
    (CommonHFKeys.NUM_ATTENTION_HEADS_UNIFORM, None),
    (CommonHFKeys.NUM_HIDDEN_LAYERS, None),
    (CommonHFKeys.VOCAB_SIZE, None),
    (CommonHFKeys.TYPE_VOCAB_SIZE, None),
    (CommonHFKeys.MAX_POSITION_EMBEDDINGS, None),
]

HF_SPECIFIC_CONFIG = HFSpecificConfig(architectures=["BertModel"], model_type="bert")


def _config_from_hf(hf_config: Mapping[str, Any], **kwargs) -> BERTConfig:
    _kwargs = config_from_hf("BERT", hf_config, HF_CONFIG_KEYS)
    _kwargs.update(kwargs)
    return BERTConfig(
        embedding_width=CommonHFKeys.HIDDEN_SIZE.get_kwarg(_kwargs),
        model_max_length=CommonHFKeys.MAX_POSITION_EMBEDDINGS.get_kwarg(_kwargs),
        **_kwargs,
    )


def _config_to_hf(curated_config: BERTConfig) -> Dict[str, Any]:
    out = config_to_hf(curated_config, [k for k, _ in HF_CONFIG_KEYS])
    return HF_SPECIFIC_CONFIG.merge(out)