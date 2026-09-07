# Final paper asset builder

Run from the repository root:

```bash
.venv/bin/python analysis/paper_assets/build_paper_assets.py
```

The builder is offline-only. It reads existing CSV/JSON/image/PDF/SVG files,
creates deterministic table summaries and figure compositions, and writes the
Prompt-defined package under `paper_assets/`.

It does not train, evaluate, alter checkpoints, alter the split, regenerate
PLAMD images, or overwrite the existing `outputs/paper_figures_v1` sources.
Figure 3 is deliberately emitted as a `MISSING_DATA` note because the complete
seven-degradation baseline restored-image grid is not present in the repository.
