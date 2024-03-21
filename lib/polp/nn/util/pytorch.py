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

from dataclasses import dataclass
from typing import Callable, List, Optional

from torch.nn import Module


@dataclass
class ModuleIterator:
    """
    Represents the details of a module when travesing a PyTorch module graph.

    :param module:
        Current module.
    :param name:
        Name of the module.
    :param prefix:
        Current dot path of the module. Includes the name.
    :param parent:
        Parent module. Will be `None` for the root module.
    """

    module: Module
    name: str
    prefix: str
    parent: Optional[Module]


def apply_to_module(module: Module, func: Callable[[ModuleIterator], None]):
    """
    Apply a function the module and its submodules in a breadth-first
    fashion.

    :param module:
        Root module.
    :param func:
        A callable that takes a module iterator as its argument.
    """
    queue: List[ModuleIterator] = [ModuleIterator(module, "", "", None)]

    while queue:
        itr = queue.pop(0)
        func(itr)

        for name, child in itr.module.named_children():
            child_prefix = f"{itr.prefix}.{name}" if itr.prefix else name
            queue.append(ModuleIterator(child, name, child_prefix, itr.module))