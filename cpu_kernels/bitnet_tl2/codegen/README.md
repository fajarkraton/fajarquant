# Vendored microsoft/BitNet `codegen_tl2.py`

This directory ships `codegen_tl2.py` (vendored 2026-04-29 from
`microsoft/BitNet @ main`, MIT) with one fajarquant-owned patch:
adds a `fajarquant_mini` entry to `ModelShapeDict` covering the
Phase D Mini ckpt's BitLinear shapes.

## To regenerate the kernel header

```bash
cd cpu_kernels/bitnet_tl2/codegen
python3 codegen_tl2.py \
  --model fajarquant_mini \
  --BM 64,64,64,64 \
  --BK 128,128,128,128 \
  --bm 32,32,32,32

# Output lands in `../include/bitnet-lut-kernels.h`. Copy it next
# to the wrapper, applying our standing patches:
cp ../include/bitnet-lut-kernels.h ../bitnet-lut-kernels-tl2.h
sed -i 's|#include "ggml-bitnet.h"|#include "ggml-bitnet-stub.h"|' \
  ../bitnet-lut-kernels-tl2.h
```

After the simple `sed` substitution above, manually re-apply the
F.11.4(b).1 BITNET_OMIT_TRANSFORM gate:

1. Insert `#if !defined(BITNET_OMIT_TRANSFORM)` (and the
   explanatory comment block) immediately BEFORE the
   `void ggml_bitnet_transform_tensor(...)` definition near the
   end of the header.
2. Insert `#endif  // !defined(BITNET_OMIT_TRANSFORM)` immediately
   AFTER the closing `}` of `ggml_bitnet_transform_tensor`.

Reason: that single function calls `aligned_malloc` →
`posix_memalign`/`free`, the only libc symbols in the file. The
gate excludes it from `-ffreestanding -nostdlib` builds (FajarOS
Nova kernel) without affecting the active FFI surface
(`fjq_tl2_qgemm_lut`, `fjq_tl2_preprocessor`,
`fjq_tl2_self_test`), which never call into it. See
`docs/FJQ_PHASE_F_F11_4B_INTEGRATION_AUDIT.md` §3.

`include/` is `.gitignore`d — it's a build artifact, not a vendored
file. Only the codegen source + `bitnet-lut-kernels-tl2.h` (the
copy) are tracked.

## To add a new shape

1. Edit `codegen_tl2.py`, find `ModelShapeDict["fajarquant_mini"]`,
   append the new `[m, k]` pair.
2. Update the BM/BK/bm comma-lists in the regen command above to
   match the new entry count.
3. Rerun. New shape-specific `three_qgemm_lut_<m>_<k>` /
   `two_qgemm_lut_<m>_<k>` functions emit automatically.
4. Add the new `(m, k)` pair to `TL2_SUPPORTED_SHAPES` in
   `src/cpu_kernels/tl2.rs`.
5. `cargo test --features bitnet_tl2 cpu_kernels` to verify.

## Constraints (per upstream)

- `M % BM == 0`
- `(K % BK) % 32 == 0`
- `bm` must be exactly `32` (assertion in `codegen_tl2.py`)

## License

`codegen_tl2.py` itself is MIT (Microsoft Corporation); see
`../MIT_LICENSE_BITNET`. The fajarquant patch (the
`fajarquant_mini` entry) is Apache-2.0 (fajarquant) per the project
LICENSE.
