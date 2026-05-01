# Third-Party Notices

This file tracks open-source code vendored into the fajarquant
repository under licenses other than fajarquant's own Apache-2.0.

---

## microsoft/BitNet TL2 x86 kernel

**License:** MIT (see `cpu_kernels/bitnet_tl2/MIT_LICENSE_BITNET`)
**Copyright:** Microsoft Corporation
**Upstream:** https://github.com/microsoft/BitNet
**Vendored at:** `cpu_kernels/bitnet_tl2/`
**Vendored on:** 2026-04-29 (V32-prep F.11.1)
**Files vendored:**
  - `cpu_kernels/bitnet_tl2/bitnet-lut-kernels-tl2.h` — preset AVX2
    kernel for the bitnet_b1_58-3B model shape. Modified ONLY to
    redirect the include path: `#include "ggml-bitnet.h"` →
    `#include "ggml-bitnet-stub.h"`. No other modifications.
  - `cpu_kernels/bitnet_tl2/kernel_config_tl2.ini` — tile parameter
    config (`bm`, `bk`, `bmm`) for the same preset shape. Verbatim.
  - `cpu_kernels/bitnet_tl2/MIT_LICENSE_BITNET` — verbatim copy of
    the upstream `LICENSE` file (SHA `9e841e7a`, 1141 bytes).

**Files NOT vendored** (referenced by upstream but stubbed locally):
  - `include/ggml-bitnet.h` — replaced by fajarquant-owned
    `cpu_kernels/bitnet_tl2/ggml-bitnet-stub.h` (Apache-2.0)
    providing only the type declarations (`bitnet_float_type`,
    `bitnet_tensor_extra`, opaque `ggml_tensor`, opaque
    `ggml_type`) that the preset header references at compile time.
    The full GGML header surface (~10K LOC) is intentionally NOT
    pulled in.

**Why this vendor strategy:** Per
`docs/FJQ_PHASE_F_F11_BITNETCPP_TL2_PORT_DESIGN.md` v1.0 §3.1
choice (b), keep upstream LOC delta minimal. The preset kernel's
public entry points (`ggml_qgemm_lut`, `ggml_preprocessor`) are
pure-`void*` / `int` signatures and don't require the full GGML
graph machinery; stubbing the type-decl-only surface lets us
compile cleanly without a 10K-LOC dependency drag.

**Attribution requirement:** Per the MIT license, redistribution
of the vendored source preserves the copyright notice. Both this
file and `cpu_kernels/bitnet_tl2/MIT_LICENSE_BITNET` satisfy that.

**Upstream tracking:** When a new BitNet release lands, re-vendor
by overwriting `bitnet-lut-kernels-tl2.h` (with the include-path
patch re-applied) and `kernel_config_tl2.ini`. The stub header is
fajarquant-owned and does not need re-sync. F.11.x follow-up may
move to `utils/codegen_tl2.py`-driven shape-specific generation
once F.11.3 parity tests pass against the 3B preset shape.

---

## AAzdi/Sparse-BitNet — N:M sparsity mask creator (Triton)

**License:** MIT (see `python/phase_d/intllm/sparse_kernel_LICENSE.txt`)
**Copyright:** (c) 2025 (per upstream LICENSE; no individual author
attribution in upstream LICENSE file)
**Upstream:** https://github.com/AAzdi/Sparse-BitNet
**Vendored at:** `python/phase_d/intllm/sparse_kernel.py`
**Vendored on:** 2026-05-01 (V32-prep F.10.1)
**Files vendored:**
  - `python/phase_d/intllm/sparse_kernel.py` — verbatim copy of
    upstream `llm/kernel/mask_creator_kernel.py` (385 LOC of Triton
    kernels + PyTorch fallback for N:M structured sparsity mask
    generation). Modifications: a 17-line `# F.10.1` attribution
    block prepended ABOVE the original module docstring; original
    code below the header is unchanged. Future modifications below
    the header MUST tag with `F.10.*` phase + commit reference.
  - `python/phase_d/intllm/sparse_kernel_LICENSE.txt` — verbatim copy
    of upstream LICENSE file (1056 bytes).

**Why vendor (not pip-install):** Sparse-BitNet is a research codebase,
not a published pip package. Vendoring keeps the kernel under our
hash-pinned versioning and allows targeted modifications without
forking upstream if Phase D HGRN-Bit shape requirements diverge.

**Phase F context:** Used by F.10 chain (GPU 2:4 structured sparsity
via Sparse-BitNet recipe). Per `docs/FJQ_PHASE_F_F10_PRODUCTION_PLAN_V0.md`
sub-task F.10.1: vendor + 2:4 invariant unit test. (Apex install
prerequisite from upstream README REMOVED 2026-05-01 — Phase D uses
pure FP32 training, no Apex needed; verified by `grep -rn "apex\|amp\|
GradScaler" python/phase_d/` returning zero hits + Mini PoL smoke
running clean without Apex.)

**Attribution requirement:** Per MIT, redistribution preserves the
copyright notice. Both this entry and the LICENSE archive satisfy that.

**Upstream tracking:** Sparse-BitNet repo had 13 stars + last commit
2026-04-22 at vendor time. Re-vendor when upstream releases material
kernel improvements; check changelog before applying.

---

*Last updated: 2026-05-01 (V32-prep F.10.1).*
