from pathlib import Path
from typing import Iterable

import numpy as np
from matplotlib.ticker import FuncFormatter


# Colours used across subplots
PANEL_COLORS = {
    'A': '#92C5DE',  # light blue
    'B': '#0072B2',  # dark blue
    'C': '#E76F00',  # light red/orange
    'D': '#B2182B',  # dark red
}
COLOR_IMPACT_ORANGE = '#E76F00'
COLOR_OUTPUT_BLUE = '#0072B2'

# Common paths (resolved relative to this file when available)
THIS_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = (THIS_DIR / '..' / 'data').resolve()
DEFAULT_GOLD_DIR = DEFAULT_DATA_ROOT / 'gold'
DEFAULT_UNICLASS_PATH = DEFAULT_DATA_ROOT / 'manual' / 'university_category' / 'ref_unique_institutions.csv'
DEFAULT_UOA_CODES_PATH = DEFAULT_DATA_ROOT / 'manual' / 'ref_acronyms' / 'ref2021_uoa_codes.xlsx'


def _resolve_first_existing(candidates: Iterable[Path], label: str) -> Path:
    for path in candidates:
        if Path(path).exists():
            return Path(path)
    joined = ", ".join(str(Path(p)) for p in candidates)
    raise FileNotFoundError(f"Could not find {label}. Looked in: {joined}")


def resolve_enhanced_ref_data_path(data_root: Path = DEFAULT_DATA_ROOT) -> Path:
    data_root = Path(data_root)
    return _resolve_first_existing(
        (
            data_root / "gold" / "enhanced_ref_data.parquet",
            data_root / "gold" / "enhanced_ref_data.csv",
            data_root / "final" / "enhanced_ref_data.csv",
            data_root / "final" / "enhanced_ref_data.zip",
        ),
        label="enhanced_ref_data",
    )


def resolve_outputs_concat_path(data_root: Path = DEFAULT_DATA_ROOT) -> Path:
    data_root = Path(data_root)
    return _resolve_first_existing(
        (
            data_root / "gold" / "outputs_concat_with_positive_authors.parquet",
            data_root / "gold" / "outputs_concat_with_positive_authors.csv",
            data_root / "dimensions_outputs" / "outputs_concat_with_positive_authors.csv",
        ),
        label="outputs_concat_with_positive_authors",
    )


def uoa_to_panel(u):
    """
    Map a Unit of Assessment number to its REF main panel label (A–D).
    Returns np.nan when the number is outside known ranges.
    """
    if 1 <= u <= 6:
        return 'A'
    if 7 <= u <= 12:
        return 'B'
    if 13 <= u <= 24:
        return 'C'
    if 25 <= u <= 34:
        return 'D'
    return np.nan


def percent_fmt(x, pos):
    """Format tick values expressed as fractions into integer percentages."""
    return f"{int(round(x * 100))}%"


def make_percent_formatter():
    """Convenience wrapper to build a FuncFormatter using percent_fmt."""
    return FuncFormatter(percent_fmt)


def size_from_value(v, vmin, vmax, s_min=10, s_max=500):
    """
    Compute marker size for a bubble plot, scaled between s_min and s_max.
    vmin/vmax define the value range; returns np.nan if value is missing.
    """
    if np.isnan(v):
        return np.nan
    if vmax == vmin:
        return 0.5 * (s_min + s_max)
    return s_min + (v - vmin) / (vmax - vmin) * (s_max - s_min)


def apply_mpl_defaults():
    """
    Apply shared Matplotlib defaults (font + title weight).

    Centralising this avoids ordering issues when different modules tweak
    rcParams at import time vs inside plotting functions.
    """
    import matplotlib as mpl

    mpl.rcParams["font.family"] = ["sans-serif"]
    mpl.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"]
    mpl.rcParams["axes.titleweight"] = "bold"
