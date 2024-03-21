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

from typing import Optional

from torch.nn import Module

from ..util.serde.load import TensorToParameterConverterT
from .bnb import prepare_for_quantization as bnb_prepare_for_quantization
from .bnb.config import BitsAndBytesConfig
from .quantizable import Quantizable


def prepare_module_for_quantization(
    module: Module, config: BitsAndBytesConfig
) -> Optional[TensorToParameterConverterT]:
    """
    Prepares a module for quantiazation and returns an optional callback
    to generate quantized parameter tensors.

    :param module:
        Top-level module to quantize. Should implement ``Quantizable``.
    :param config:
        Configuration for the quantizer.
    :returns:
        An optional callable that converts a non-quantized tensor
        to a quantized parameter.
    """
    if not isinstance(module, Quantizable):
        raise ValueError(f"Module of type `{type(module)}` is not quantizable")
    qmodel: Quantizable = module
    non_quantizable_module_prefixes = qmodel.modules_to_not_quantize()

    return bnb_prepare_for_quantization(module, config, non_quantizable_module_prefixes)