# Roadmap

Future work for fluxgen-cli, ordered by readiness. Items here are tracked but not yet
in active development.

---

## Krea 2 support (`--model krea2`)

**Status:** Waiting on upstream `mflux` release containing Krea 2.

**Trigger:** `filipstrand/mflux` merged Krea 2 (PR #453) on 2026-06-30, but the latest
PyPI release (`0.18.0`) shipped on 2026-06-07 — before that merge. Krea 2 is currently
only available on `main` and is not yet on PyPI.

**Wait condition:** Next `mflux` PyPI release that imports `from mflux.models.krea2 import Krea2`
cleanly (currently `>=0.19.0` candidate).

**Once available, do:**

1. `pyproject.toml`: bump `mflux>=0.18.0` to the new floor
2. `fluxgen/generator.py`:
   - Add `"krea2"` to the model registry in `fluxgen/models.py`
     (`capabilities={"generate"}`, `guidance=1.0`, `steps=8`, factory)
3. `tests/test_cli.py`: extend `load_cli_without_mflux` fixture's generation model list
4. `README.md`: document the new model, recommended `--steps 8 --quantize 8`, ~24 GB
   download warning, img2img caveat (Turbo only supports strength-based, not Krea's
   hosted style-reference path)
5. `CHANGELOG.md`: add entry
6. Smoke test: `fluxgen gen "a fox in a forest" --model krea2 --steps 8 -q 8`

**Caveats to preserve:**
- Krea 2 is **txt2img-only** upstream — do NOT expose it under `fluxgen edit`
- Existing PRESETS (5/9/16 steps) are wrong for Turbo's 8-step distillation; document
  or override when `--model krea2` is selected
- Turbo + q8 needs ~32 GB+ unified memory; do not advertise as 16 GB-friendly
- If Krea 2 Raw (non-distilled) is wanted later, that needs a new upstream initializer
  — out of scope here

**Reference:**
- Upstream PR: https://github.com/filipstrand/mflux/pull/453
- Model: https://huggingface.co/krea/Krea-2-Turbo
- Technical report: https://www.krea.ai/blog/krea-2-technical-report
