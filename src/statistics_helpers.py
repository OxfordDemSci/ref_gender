from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from figure_one_helpers import DEFAULT_DATA_ROOT
from figure_one_data import prepare_figure_one_data

ALPHA = 0.05


def load_statistics_data(data_root: Path = DEFAULT_DATA_ROOT):
    """
    Load raw ICS/output tables plus aggregated UoA/university frames.
    Returns (df_output, df_ics, df_uoa_m, df_uni_m, df_uniuoa_m).
    """
    data_root = Path(data_root)
    df_output = pd.read_csv(data_root / "dimensions_outputs/outputs_concat_with_positive_authors.csv")
    df_ics = pd.read_csv(data_root / "final/enhanced_ref_data.csv")
    df_ics, df_uoa_m, df_uni_m, df_uniuoa_m = prepare_figure_one_data(data_root)
    return df_output, df_ics, df_uoa_m, df_uni_m, df_uniuoa_m


def _safe_pct(num: float, denom: float) -> float:
    return 100 * num / denom if denom else np.nan


def build_descriptive_summary(df_ics: pd.DataFrame, df_uoa_m: pd.DataFrame, df_uni_m: pd.DataFrame, df_output: pd.DataFrame) -> str:
    """Generate a multiline descriptive summary."""
    lines = []
    lines.append("=" * 60)
    lines.append("DESCRIPTIVE SUMMARY OF FEMALE REPRESENTATION IN ICS & OUTPUTS")
    lines.append("=" * 60 + "\n")

    n_fem_out = df_output["number_female"].sum()
    n_male_out = df_output["number_male"].sum()
    n_fem_ics = df_ics["number_female"].sum()
    n_male_ics = df_ics["number_male"].sum()
    pct_fem_out = _safe_pct(n_fem_out, n_fem_out + n_male_out)
    pct_fem_ics = _safe_pct(n_fem_ics, n_fem_ics + n_male_ics)

    lines.append("Overall female share:")
    lines.append(f"  • Outputs: {n_fem_out:,} women / {n_fem_out + n_male_out:,} total = {pct_fem_out:.2f}% female")
    lines.append(f"  • ICS:     {n_fem_ics:,} women / {n_fem_ics + n_male_ics:,} total = {pct_fem_ics:.2f}% female\n")

    # All-women ICS
    ics_all_female = df_ics[(df_ics["number_male"] == 0) & (df_ics["number_female"] > 0)]
    n_all_fem_ics = len(ics_all_female)
    pct_all_fem_ics = _safe_pct(n_all_fem_ics, len(df_ics))
    lines.append(f"All-female ICS submissions (excluding unknowns): {n_all_fem_ics:,} ({pct_all_fem_ics:.2f}% of all ICS cases)\n")

    # By University
    top_uni = df_uni_m[["Institution name", "pct_female_ics", "pct_female_output"]].dropna().sort_values("pct_female_ics", ascending=False)
    bottom_uni = top_uni.sort_values("pct_female_ics", ascending=True)

    lines.append("Universities with highest female Impact (ICS) proportions:")
    lines.append(top_uni.head(5).to_string(index=False))
    lines.append("\nUniversities with lowest female Impact (ICS) proportions:")
    lines.append(bottom_uni.head(5).to_string(index=False))

    df_uni_m = df_uni_m.copy()
    df_uni_m["diff_ics_output"] = df_uni_m["pct_female_ics"] - df_uni_m["pct_female_output"]
    lines.append("\nUniversities with largest positive difference (ICS − Output):")
    lines.append(
        df_uni_m[["Institution name", "pct_female_ics", "pct_female_output", "diff_ics_output"]]
        .sort_values("diff_ics_output", ascending=False)
        .head(5)
        .to_string(index=False)
    )
    lines.append("\nUniversities with largest negative difference (ICS − Output):")
    lines.append(
        df_uni_m[["Institution name", "pct_female_ics", "pct_female_output", "diff_ics_output"]]
        .sort_values("diff_ics_output", ascending=True)
        .head(5)
        .to_string(index=False)
    )

    # By UoA
    df_uoa_m = df_uoa_m.copy()
    df_uoa_m["diff_ics_output"] = df_uoa_m["pct_female_ics"] - df_uoa_m["pct_female_output"]
    top_uoa = df_uoa_m[["Unit of assessment name", "pct_female_ics", "pct_female_output"]].dropna().sort_values("pct_female_ics", ascending=False)
    bottom_uoa = top_uoa.sort_values("pct_female_ics", ascending=True)

    lines.append("\nUoAs with highest female Impact (ICS) proportions:")
    lines.append(top_uoa.head(5).to_string(index=False))
    lines.append("\nUoAs with lowest female Impact (ICS) proportions:")
    lines.append(bottom_uoa.head(5).to_string(index=False))
    lines.append("\nUoAs with largest positive difference (ICS − Output):")
    lines.append(
        df_uoa_m[["Unit of assessment name", "pct_female_ics", "pct_female_output", "diff_ics_output"]]
        .sort_values("diff_ics_output", ascending=False)
        .head(5)
        .to_string(index=False)
    )
    lines.append("\nUoAs with largest negative difference (ICS − Output):")
    lines.append(
        df_uoa_m[["Unit of assessment name", "pct_female_ics", "pct_female_output", "diff_ics_output"]]
        .sort_values("diff_ics_output", ascending=True)
        .head(5)
        .to_string(index=False)
    )

    # Discipline groups (STEM vs SHAPE)
    disc = (
        df_uoa_m[["Discipline_group", "pct_female_ics", "pct_female_output"]]
        .dropna(subset=["Discipline_group", "pct_female_ics", "pct_female_output"])
        .groupby("Discipline_group")[["pct_female_ics", "pct_female_output"]]
        .mean()
    )
    disc["diff"] = disc["pct_female_ics"] - disc["pct_female_output"]
    lines.append("\nAverage female share by Discipline group (STEM vs SHAPE):")
    lines.append(disc.to_string(float_format=lambda x: f"{x:.3f}"))

    # Distribution summaries
    lines.append("\nSummary statistics of female proportions across Universities and UoAs:")
    lines.append("Universities (Impact):")
    lines.append(df_uni_m["pct_female_ics"].describe().round(3).to_string())
    lines.append("\nUoAs (Impact):")
    lines.append(df_uoa_m["pct_female_ics"].describe().round(3).to_string())
    lines.append("\nUniversities (Outputs):")
    lines.append(df_uni_m["pct_female_output"].describe().round(3).to_string())
    lines.append("\nUoAs (Outputs):")
    lines.append(df_uoa_m["pct_female_output"].describe().round(3).to_string())

    return "\n".join(lines)


def _wald_ci_proportion(p, n, alpha=ALPHA):
    if n <= 0 or np.isnan(p):
        return (np.nan, np.nan)
    z = stats.norm.ppf(1 - alpha / 2)
    se = np.sqrt(p * (1 - p) / n)
    return (p - z * se, p + z * se)


def _wald_ci_diff(p1, n1, p2, n2, alpha=ALPHA):
    if min(n1, n2) <= 0:
        return (np.nan, np.nan)
    z = stats.norm.ppf(1 - alpha / 2)
    se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    diff = p1 - p2
    return (diff - z * se, diff + z * se)


def _two_prop_ztest(count1, nobs1, count2, nobs2, alternative="larger", alpha=ALPHA) -> Dict:
    p1 = count1 / nobs1
    p2 = count2 / nobs2
    p_pool = (count1 + count2) / (nobs1 + nobs2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / nobs1 + 1 / nobs2))
    z = (p1 - p2) / se
    if alternative == "two-sided":
        pval = 2 * (1 - stats.norm.cdf(abs(z)))
        alt_text = "p1 ≠ p2"
    elif alternative == "larger":
        pval = 1 - stats.norm.cdf(z)
        alt_text = "p1 > p2"
    else:
        pval = stats.norm.cdf(z)
        alt_text = "p1 < p2"
    ci_lo, ci_hi = _wald_ci_diff(p1, nobs1, p2, nobs2, alpha=alpha)
    signif = pval < alpha
    return {
        "z": z,
        "p": pval,
        "diff": p1 - p2,
        "ci": (ci_lo, ci_hi),
        "alt_text": alt_text,
        "significant": signif,
    }


def _describe_prop(name: str, p: float, n: int, alpha=ALPHA) -> str:
    lo, hi = _wald_ci_proportion(p, n, alpha)
    return f"{name}: p̂ = {p:.4f}  (95% CI [{lo:.4f}, {hi:.4f}]), n = {n}"


def _paired_suite(label: str, x_ics: pd.Series, x_out: pd.Series, alpha=ALPHA) -> str:
    df = pd.DataFrame({"ics": x_ics, "out": x_out}).dropna()
    d = (df["ics"] - df["out"]).to_numpy()
    n = d.size
    if n == 0:
        return f"{label}: no paired observations."
    mean_d = d.mean()
    sd_d = d.std(ddof=1) if n > 1 else np.nan
    z = stats.norm.ppf(1 - alpha / 2)
    se = (sd_d / np.sqrt(n)) if n > 1 else np.nan
    ci = (mean_d - z * se, mean_d + z * se) if n > 1 else (np.nan, np.nan)
    dz = mean_d / sd_d if (n > 1 and sd_d > 0) else np.nan

    lines = []
    lines.append(f"\n— {label}: paired analysis for Δ = (ICS − Output)")
    lines.append(f"  Descriptives: n = {n}, mean(Δ) = {mean_d:.4f} (95% CI [{ci[0]:.4f}, {ci[1]:.4f}]), sd = {sd_d:.4f}, Cohen's dz = {dz:.3f}")

    if n >= 2:
        t_stat, p_t = stats.ttest_1samp(d, 0.0, alternative="greater")
        signif_t = "significant" if p_t < alpha else "not significant"
        direction_t = "positive" if mean_d > 0 else ("negative" if mean_d < 0 else "zero")
        lines.append(f"  t-test (mean Δ > 0): t = {t_stat:.3f}, p = {p_t:.3g} → {signif_t} at α={alpha} (mean Δ {direction_t}).")
    else:
        lines.append("  t-test: not applicable (n<2).")

    d_nz = d[d != 0]
    if d_nz.size > 0:
        try:
            w_stat, p_w = stats.wilcoxon(d_nz, alternative="greater", zero_method="wilcox", correction=False)
            med = np.median(d)
            signif_w = "significant" if p_w < alpha else "not significant"
            direction_w = "positive" if med > 0 else ("negative" if med < 0 else "zero")
            lines.append(f"  Wilcoxon (median Δ > 0): W = {w_stat:.3f}, p = {p_w:.3g} → {signif_w} (median Δ {direction_w}).")
        except Exception as e:
            lines.append(f"  Wilcoxon: not computed ({e}).")
    else:
        lines.append("  Wilcoxon: all differences are zero; test not applicable.")

    n_pos = int(np.sum(d > 0))
    n_neg = int(np.sum(d < 0))
    n_eff = n_pos + n_neg
    if n_eff > 0:
        p_bin = stats.binomtest(n_pos, n_eff, p=0.5, alternative="greater").pvalue
        frac_pos = n_pos / n_eff
        signif_s = "significant" if p_bin < alpha else "not significant"
        lines.append(f"  Sign test (P(Δ>0) > 0.5): positives = {n_pos}/{n_eff} ({frac_pos:.3%}), p = {p_bin:.3g} → {signif_s}.")
    else:
        lines.append("  Sign test: no nonzero differences; not applicable.")
    return "\n".join(lines)


def build_inference_summary(
    df_ics: pd.DataFrame,
    df_output: pd.DataFrame,
    df_uoa_m: pd.DataFrame,
    df_uni_m: pd.DataFrame,
    df_uniuoa_m: Optional[pd.DataFrame] = None,
    alpha=ALPHA,
) -> str:
    """
    One-sided tests asking whether female share in ICS exceeds Outputs at multiple levels.
    If the institution×UoA frame is not provided, it is re-computed from disk.
    """
    if df_uniuoa_m is None:
        _, _, _, _, df_uniuoa_m = load_statistics_data()

    lines = []
    lines.append("Hypothesis across all levels: female proportion in ICS exceeds Outputs (one-sided tests).\n")

    x_ics = int(df_ics["number_female"].sum())
    n_ics = int((df_ics["number_female"] + df_ics["number_male"]).sum())
    x_out = int(df_output["number_female"].sum())
    n_out = int((df_output["number_female"] + df_output["number_male"]).sum())

    p_ics = x_ics / n_ics if n_ics > 0 else np.nan
    p_out = x_out / n_out if n_out > 0 else np.nan

    lines.append("RAW pooled female shares:")
    lines.append(_describe_prop("  ICS   ", p_ics, n_ics, alpha=alpha))
    lines.append(_describe_prop("  Output", p_out, n_out, alpha=alpha))

    res = _two_prop_ztest(x_ics, n_ics, x_out, n_out, alternative="larger", alpha=alpha)
    lines.append("\nTwo-proportion z-test (RAW):")
    lines.append("  H0: p_ICS = p_Output   vs   H1: p_ICS > p_Output")
    ci_lo, ci_hi = res["ci"]
    signif_text = "statistically significant" if res["significant"] else "not statistically significant"
    direction = "positive" if res["diff"] > 0 else ("negative" if res["diff"] < 0 else "zero")
    lines.append(
        f"  z = {res['z']:.3f}, p = {res['p']:.3g} (H0: p1 = p2 vs H1: {res['alt_text']}). "
        f"Observed difference p1−p2 = {res['diff']:+.4f} [{ci_lo:+.4f}, {ci_hi:+.4f}] "
        f"(95% CI, Wald, unpooled). Result is {signif_text} at α={alpha}; the estimated difference is {direction}."
    )
    lines.append("  Interpretation: This tests the overall female share across all observations. A significant result supports higher ICS share.\n")

    uni = df_uni_m[["pct_female_ics", "pct_female_output"]].dropna()
    lines.append(_paired_suite("University level", uni["pct_female_ics"], uni["pct_female_output"], alpha=alpha))
    lines.append("  Interpretation: Positive, significant results indicate universities tend to have higher female shares in ICS than in Outputs.\n")

    uoa = df_uoa_m[["pct_female_ics", "pct_female_output"]].dropna()
    lines.append(_paired_suite("Unit of Assessment (UoA) level", uoa["pct_female_ics"], uoa["pct_female_output"], alpha=alpha))
    lines.append("  Interpretation: Positive, significant results indicate disciplines (UoAs) tend to have higher female shares in ICS.\n")

    inst_uoa = df_uniuoa_m[["pct_female_ics", "pct_female_output"]].dropna()
    lines.append(_paired_suite("Institution × UoA level", inst_uoa["pct_female_ics"], inst_uoa["pct_female_output"], alpha=alpha))
    lines.append("  Interpretation: At the cell level, a significant positive Δ indicates the tendency persists at finer granularity.\n")

    lines.append("RECAP:")
    lines.append("  • RAW z-test asks: Is the *overall* female share higher in ICS than in Outputs?")
    lines.append("  • Paired tests (University, UoA, and Inst×UoA) ask: Within each unit, is ICS share higher than Outputs (Δ>0)?")
    lines.append("  • Convergent significance across multiple tests provides robust evidence that female ICS contribution exceeds Output.")

    return "\n".join(lines)
