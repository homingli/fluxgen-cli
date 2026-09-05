# Roadmap

Future work for fluxgen-cli, ordered by readiness. Items here are tracked but not yet
in active development.

---

## Recently completed

- **Krea 2 support (`--model krea2`)** — shipped. `krea2` (Krea 2 Turbo) is registered as
  a generation-only model (8-step distilled, CFG 1.0); `mflux` floor is `>=0.19.1`
  (Krea 2 landed upstream in 0.18.1). Recommended usage, download/memory caveats, and
  the strength-based img2img note live in `README.md` under the generation docs.
  Krea 2 Raw (non-distilled) remains unsupported — it needs a new upstream initializer.
