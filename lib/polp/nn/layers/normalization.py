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

import torch
from torch import Tensor
from torch.nn import Module, Parameter


class RMSNorm(Module):
    """
    Root Mean Square (RMS) normalization (`Zhang et al., 2019`_).

    .. _Zhang et al., 2019: https://arxiv.org/abs/1910.07467
    """

    def __init__(
        self, width: int, *, eps: float, device: Optional[torch.device] = None
    ):
        """
        Construct a RMS normalization module.

        :param width:
            The (hidden) width of the representations that RMS
            normalization will be applied to.
        :param eps:
            Epsilon to avoid division by zero.
        :param device:
            Device on which the module is to be initialized.
        """
        super().__init__()
        self.eps = eps
        self.weight = Parameter(torch.ones((width,), device=device))

    def forward(self, input: Tensor) -> Tensor:
        """
        Apply RMS normalization to a tensor.

        :param input:
            The tensor to apply normalization to.
        :returns:
            Normalized tensor.
        """
        # Zhang & Sennrich, Equation 4. If we are in lower precision than
        # float32, then squaring and averaging can get way off. So for
        # normalization we want to use higher precision.
        rms = (
            input.to(torch.float32)
            .square()
            .mean(-1, keepdim=True)
            .add(self.eps)
            .rsqrt()
        )

        return (input * rms).to(input.dtype) * self.weight