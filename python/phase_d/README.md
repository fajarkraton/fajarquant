# IntLLM — FajarQuant Phase D

Fully integer-native language model. Research artifact for the V31
Track C work described in `fajarquant/docs/FJQ_PHASE_D_*.md`.

**Status:** C.P2 scaffolding — not yet trained. See `docs/FJQ_PHASE_D_P2_FINDINGS.md`.

## Layout

```
python/phase_d/
├── pyproject.toml       # intllm package (this project)
├── intllm/              # source code
│   ├── model.py         # MLGRU + GLU + BitLinear
│   ├── quant.py         # FajarQuant QAT recipe
│   ├── train.py         # training loop
│   ├── data.py          # SlimPajama streaming loader
│   ├── eval.py          # Wikitext-103 + LAMBADA + MMLU
│   └── export.py        # .fjm v9 for FajarOS Nova
├── tests/               # pytest unit tests
├── scripts/             # repro_upstream.py, train_*.py, cleanup_hf_cache.py
├── configs/             # hparam YAML per Mini/Base/Medium/Stretch
└── _upstream/           # vendored snapshot of ridgerchu/matmulfreellm (frozen-dep)
```

## Install (dev)

```bash
cd fajarquant/python/phase_d
python3 -m venv .venv && source .venv/bin/activate  # or reuse fajarquant/.venv
pip install -e ".[train,eval,dev]"
```

Reuses `fajarquant/.venv` by default. Torch 2.6.0+cu124 already
present in that venv.

## Reference papers

- [MatMul-Free LLM](https://arxiv.org/abs/2406.02528) — primary arch
- [BitNet b1.58](https://arxiv.org/abs/2402.17764) — ternary weight recipe
- [RWKV-7 "Goose"](https://arxiv.org/abs/2503.14456) — secondary ablation

## License

Apache-2.0 (inherits from `fajarquant` parent).
