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

from typing import TYPE_CHECKING

from ..util.serde.checkpoint import ModelCheckpointType

HF_MODEL_CONFIG = "config.json"
HF_MODEL_CHECKPOINT = "pytorch_model.bin"
HF_MODEL_CHECKPOINT_SAFETENSORS = "model.safetensors"
HF_MODEL_SHARDED_CHECKPOINT_INDEX = "pytorch_model.bin.index.json"
HF_MODEL_SHARDED_CHECKPOINT_INDEX_SAFETENSORS = "model.safetensors.index.json"
HF_MODEL_SHARDED_CHECKPOINT_INDEX_WEIGHTS_KEY = "weight_map"
HF_TOKENIZER_CONFIG = "tokenizer_config.json"
SPECIAL_TOKENS_MAP = "special_tokens_map.json"
TOKENIZER_JSON = "tokenizer.json"


PRIMARY_CHECKPOINT_FILENAMES = {
    ModelCheckpointType.PYTORCH_STATE_DICT: HF_MODEL_CHECKPOINT,
    ModelCheckpointType.SAFE_TENSORS: HF_MODEL_CHECKPOINT_SAFETENSORS,
}
SHARDED_CHECKPOINT_INDEX_FILENAMES = {
    ModelCheckpointType.PYTORCH_STATE_DICT: HF_MODEL_SHARDED_CHECKPOINT_INDEX,
    ModelCheckpointType.SAFE_TENSORS: HF_MODEL_SHARDED_CHECKPOINT_INDEX_SAFETENSORS,
}
# Same for both checkpoint types.
SHARDED_CHECKPOINT_INDEX_WEIGHTS_KEY = HF_MODEL_SHARDED_CHECKPOINT_INDEX_WEIGHTS_KEY