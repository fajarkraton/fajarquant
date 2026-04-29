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
# to the wrapper, applying our standing include patch:
cp ../include/bitnet-lut-kernels.h ../bitnet-lut-kernels-tl2.h
sed -i 's|#include "ggml-bitnet.h"|#include "ggml-bitnet-stub.h"|' \
  ../bitnet-lut-kernels-tl2.h
```

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
