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

import functools
from contextlib import contextmanager
from typing import List

import torch
from torch.nn import Module
from torch.utils.hooks import RemovableHandle

from .pytorch import ModuleIterator, apply_to_module


@contextmanager
def use_nvtx_ranges_for_forward_pass(module: Module):
    """
    Recursively applies `NVTX ranges`_ to the forward pass operation
    of the provided module. The ranges will be recorded during an `Nsight`_
    profiling session.

    :param module:
        Top-level module to which the ranges are applied recursively.

    .. _Nsight: https://developer.nvidia.com/nsight-systems
    .. _NVTX ranges: https://pytorch.org/docs/stable/cuda.html#nvidia-tools-extension-nvtx
    """

    hooks: List[RemovableHandle] = []

    def hook_forward(itr: ModuleIterator):
        range_name = f"{itr.name} : {type(itr.module).__name__}"

        def push(*args, _range_name: str, **kwargs):
            torch.cuda.nvtx.range_push(_range_name)

        def pop(*args, **kwargs):
            torch.cuda.nvtx.range_pop()

        forward_pre = itr.module.register_forward_pre_hook(
            functools.partial(push, _range_name=range_name)
        )
        forward_post = itr.module.register_forward_hook(pop)
        hooks.append(forward_pre)
        hooks.append(forward_post)

    try:
        apply_to_module(module, hook_forward)
        yield
    finally:
        for hook in hooks:
            hook.remove()