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

import warnings
from importlib.util import find_spec

_has_scipy = find_spec("scipy") is not None and find_spec("scipy.stats") is not None
has_bitsandbytes = find_spec("bitsandbytes") is not None

# As of v0.40.0, `bitsandbytes` doesn't correctly specify `scipy` as an installation
# dependency. This can lead to situations where the former is installed but
# the latter isn't and the ImportError gets masked. So, we additionally check
# for the presence of `scipy`.
if has_bitsandbytes and not _has_scipy:
    warnings.warn(
        "The `bitsandbytes` library is installed but its dependency "
        "`scipy` isn't. Please install `scipy` to correctly load `bitsandbytes`."
    )
    has_bitsandbytes = False

has_safetensors = find_spec("safetensors")