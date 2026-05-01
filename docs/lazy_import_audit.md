# Lazy import audit

256 import statements live inside function bodies across the repo. They fall into four categories:

| Category | Count | Action |
|---|---|---|
| GPU / RAPIDS optional deps (cudf, cugraph, cuopt, cupy, rmm, ...) | ~25 | **Keep lazy.** Loading these at module import would break on machines without RAPIDS. |
| Isaac Sim optional deps (omni, isaacsim, pxr, visualization_isaac.*) | ~23 | **Keep lazy.** Only available inside the Isaac Sim conda env. |
| Heavy third-party libs that should stay deferred (matplotlib, open3d, plotly, trimesh, scipy.ndimage, numba) | ~18 | **Keep lazy.** Significant import cost; only paid when the visualisation/post-processing path runs. |
| In-function imports of first-party modules (run_full_pipeline.py stages, experiments) | ~190 | **Mostly fine.** These are not lazy in the optimisation sense -- they're stage-by-stage imports inside long `main()` functions, which keeps each stage's dependencies visible at the call site. Promoting them to top-level would be cosmetic only. |

## Verdict

No code change in this pass. Recommendation:
- The first three categories are load-bearing -- moving them up would break things.
- The fourth category is style; if we ever want to clean it up, restrict to **scripts entry points only** (run_full_pipeline.py, VRP/scripts/run_vrp.py, experiments/e0*.py) and only for the imports of plain first-party modules. Skip imports near `try:`/`except ImportError:` guards.

## How to regenerate

The categorisation lived in a one-off Python snippet. Re-run with:

```python
import os
LAZY = []
for root, _, files in os.walk('.'):
    if any(x in root for x in ['.git', '__pycache__', 'data', 'plots']):
        continue
    for f in files:
        if not f.endswith('.py'):
            continue
        path = os.path.join(root, f)
        for i, line in enumerate(open(path).readlines(), 1):
            stripped = line.lstrip()
            if (stripped.startswith('import ') or stripped.startswith('from ')) and len(line) > len(stripped):
                LAZY.append((path, i, stripped.rstrip()))
```

Then bucket by keywords (`cupy`, `cudf`, `cugraph`, `cuopt`, `omni`, `matplotlib`, `open3d`, ...) to reproduce the table.
