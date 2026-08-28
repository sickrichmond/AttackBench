# Third-Party Notices

AttackBench is distributed under the MIT License. See [LICENSE](LICENSE).
That license covers rights held by AttackBench contributors; the components
listed below retain their upstream copyrights and license conditions.

The final sections separately document optional external checkpoints that are not
included in the AttackBench source distribution or wheel.

## AutoAttack (MIT)

Files:

- `attackbench/attacks/original/auto_pgd.py`
- `attackbench/attacks/original/fast_adaptive_boundary.py`

Source: <https://github.com/fra31/auto-attack>

Copyright (c) 2020 Francesco Croce

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## CROWN-IBP model definitions (BSD-2-Clause)

File:

- `attackbench/models/original/zhang2020/model_defs_gowal.py`

Source: <https://github.com/huanzhang12/CROWN-IBP>

Copyright (c) 2019 Huan Zhang, Hongge Chen and Chaowei Xiao
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## Optional external checkpoint: Stutz 2020 CCAT (noncommercial)

AttackBench contains an independently implemented MIT architecture compatible with this
checkpoint. The checkpoint and upstream implementation are not included in AttackBench.
If a user explicitly accepts the upstream terms, AttackBench can download the checkpoint
directly from the author's server. The following upstream notice applies to that external
asset:

Copyright (c) 2020 David Stutz, Max-Planck-Gesellschaft

Please read carefully the following terms and conditions and any accompanying
documentation before you download and/or use this software and associated documentation
files (the "Software").

The authors hereby grant you a non-exclusive, non-transferable, free of charge right to
copy, modify, merge, publish, distribute, and sublicense the Software for the sole purpose
of performing non-commercial scientific research, non-commercial education, or
non-commercial artistic projects.

Any other use, in particular any use for commercial purposes, is prohibited. This
includes, without limitation, incorporation in a commercial product, use in a commercial
service, or production of other artefacts for commercial purposes.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.

You understand and agree that the authors are under no obligation to provide either
maintenance services, update services, notices of latent defects, or corrections of
defects with regard to the Software. The authors nevertheless reserve the right to
update, modify, or discontinue the Software at any time.

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software. You agree to cite the corresponding papers in
documents and papers that report on research using the Software.

Source and complete current terms:
<https://github.com/davidstutz/confidence-calibrated-adversarial-training#license>

## Optional external checkpoint: Xiao 2020 k-WTA (no stated license)

AttackBench contains an independently implemented MIT architecture compatible with the
published k-WTA checkpoint. Neither the upstream source nor its checkpoint is included or
automatically downloaded because the upstream repository does not state a license. Users
must obtain the checkpoint separately and are responsible for establishing the rights
needed for their use.

Source: <https://github.com/wielandbrendel/robustness_workshop>
