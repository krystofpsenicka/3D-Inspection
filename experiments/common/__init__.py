from .config import ModelConfig, ExperimentConfig, SEEDS_3, SEEDS_5, SEEDS_10
from .plotting import setup_thesis_style, save_figure
from .persistence import save_run_result, load_run_result, aggregate_to_csv
from .stats import mean_ci, format_mean_std

# `runner` pulls in cupy/RMM, so it is intentionally NOT eagerly imported
# here. Scripts that need it use `from experiments.common.runner import ...`
# directly. Keeping it lazy lets `--plots_only` runs work in a plain
# matplotlib+numpy environment.
