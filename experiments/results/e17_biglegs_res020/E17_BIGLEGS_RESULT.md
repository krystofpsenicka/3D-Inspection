# E17 legs — bigger-legs scaling benchmark (2026-08-20)

**Question:** the production-resolution legs arm found the CPU A\* competitive
with (even faster than) `gpu_pf` per leg. Is that a genuine result, or just an
artifact of Duke's individual A\* legs being tiny? This re-runs the legs arm on
an **8× finer MAPF grid** so each search is much larger.

## Method
`--arm legs` on Duke, K=5, 3 seeds (42/123/7), 20 intercepted legs each = **60
legs**, but with `SPACE_TIME_RESOLUTION = 0.20` instead of the production 0.50.
That forces factor-1 downsampling → the MAPF coarse grid is the **full fine grid
(317×113×106, ~8× more cells)**, `T=4000`, so per-leg searches are much bigger.
This is a scaling probe, **not** the production config. Data:
[`biglegs_from_logs.json`](biglegs_from_logs.json).

> Reproduce: `SPACE_TIME_RESOLUTION=0.20` must reach `VRP/core/constants.py`
> (temporarily make line 24 read from the env, or set the constant to 0.20), then
> `python -m experiments.e17_stastar_ablation --arm legs --output_dir <dir>
> --max_legs 20 --seeds 42` per seed. The fine-grid **full mission** OOMs a 24 GB
> GPU in the post-interception tail (a 15.8 GB `g_cost`/`pred` allocation — the
> exact case the `space_time_search.py` T_local memory-guard WIP caps); the 20
> intercepted legs per seed all complete first, so they were salvaged from the
> run logs. Numbers here are those 60 legs.

## Result (60 big legs)

| variant | mean (s) | median (s) | max (s) |
|---|---|---|---|
| `gpu_pf`  | 0.108 | 0.090 | **0.416** |
| `gpu_seq` | 14.61 | 3.349 | **276.4** |
| `cpu_seq` | 0.170 | 0.043 | **3.217** |

| ratio | mean | median | max |
|---|---|---|---|
| parallel-frontier `gpu_seq / gpu_pf` | 78.1× | **38.0×** | 665× |
| GPU-vs-CPU `cpu_seq / gpu_pf` (>1 = GPU faster) | 0.92× | 0.46× | — |

- `gpu_pf` beats `cpu_seq` on **20 / 60** legs.
- 3 legs are genuinely hard (`cpu_seq > 0.5 s`): there `gpu_pf` averages 0.34 s
  vs `cpu_seq` 1.82 s → **GPU 5× faster**.
- Realised cost inflation `cost_pf / cost_seq` = **1.0000** on every leg (same as
  production — the `f_delta=2.0` relaxation is free).

## Verdict
Two separable claims, and a finer grid sharpens both:

1. **Parallel frontier vs sequential GPU — decisive, and grows with size.**
   `gpu_pf` is 38× faster than `gpu_seq` at the median (vs 33× at production
   size), up to 665×; `gpu_seq` degrades catastrophically (276 s worst leg).
   This is the C1 contribution and it is unambiguous.

2. **GPU vs a competent CPU A\* — only on the hard tail.** The *median* leg still
   favors the CPU (0.46×) because even fine-grid Duke legs are mostly small and
   the GPU's ~0.03–0.1 s launch overhead dominates. But `gpu_pf`'s worst case is
   **bounded** (max 0.42 s) while the CPU spikes to 3.2 s and `gpu_seq` to 276 s.
   So the GPU's practical win is **worst-case / tail latency and robustness to
   hard queries**, not median per-leg speed — and the mean is already at parity
   (0.92×). A finer grid does not flip the median: Duke's individual legs simply
   are not big enough for the GPU to beat a near-zero-overhead CPU on the typical
   case.

This does not change the production-resolution legs conclusion; it explains it
and bounds it.
