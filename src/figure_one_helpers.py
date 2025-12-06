from pathlib import Path

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
DEFAULT_UNICLASS_PATH = DEFAULT_DATA_ROOT / 'manual' / 'university_category' / 'ref_unique_institutions.csv'


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
