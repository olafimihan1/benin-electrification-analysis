"""
Run Script: Benin Least-Cost Electrification Analysis
======================================================

This script:
1. Runs the electrification analysis (demand + cost + least-cost selection)
2. Generates summary statistics
3. Produces 6 visualizations
4. Identifies top-priority settlements

Usage:
    python3 run_analysis.py

Requires: benin_electrification.py in the same directory or on sys.path.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

# ── Ensure the output directory (where benin_electrification.py lives) is importable
OUTPUT_DIR = Path("./outputs")
sys.path.insert(0, str(OUTPUT_DIR))

from benin_electrification import (
    run as run_analysis,
    run_sensitivity,
    SENSITIVITY_PARAMS,
)

# ── Paths ──────────────────────────────────────────────────────────
DATA_FILE = Path("./data/Benin_settlement_properties.geojson")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Logging ────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── Plot style ─────────────────────────────────────────────────────
sns.set_style("whitegrid")
plt.rcParams.update({"figure.figsize": (12, 7), "font.size": 10})

TECH_COLORS = {"Grid": "#2ecc71", "Mini-grid": "#e74c3c", "SHS": "#3498db"}


# ═══════════════════════════════════════════════════════════════════
# VISUALISATION HELPERS
# ═══════════════════════════════════════════════════════════════════

def save(fig, name):
    fig.savefig(OUTPUT_DIR / name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved: {name}")


def plot_technology_distribution(df):
    """01 — Pie chart of least-cost technology split."""
    fig, ax = plt.subplots(figsize=(10, 7))
    counts = df["least_cost_technology"].value_counts()
    colors = [TECH_COLORS.get(t, "#95a5a6") for t in counts.index]
    ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%",
           colors=colors, startangle=90, textprops={"fontsize": 12})
    ax.set_title("Least-Cost Technology Distribution",
                 fontsize=14, fontweight="bold", pad=20)
    save(fig, "01_technology_distribution.png")


def plot_lcoe_comparison(df):
    """02 — Box-plot of LCOE by technology (viable settlements only)."""
    fig, ax = plt.subplots(figsize=(12, 7))
    data, labels, colors = [], [], []
    for col, label in [("grid_lcoe", "Grid"),
                       ("minigrid_lcoe", "Mini-grid"),
                       ("shs_lcoe", "SHS")]:
        vals = df[col].dropna()
        vals = vals[vals < np.inf]
        if len(vals):
            data.append(vals)
            labels.append(label)
            colors.append(TECH_COLORS.get(label, "#95a5a6"))

    bp = ax.boxplot(data, labels=labels, patch_artist=True, showmeans=True)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    ax.set_ylabel("LCOE (USD / MWh)", fontsize=12, fontweight="bold")
    ax.set_title("Levelized Cost of Electricity by Technology",
                 fontsize=14, fontweight="bold", pad=20)
    ax.grid(True, alpha=0.3)
    save(fig, "02_lcoe_comparison.png")


def plot_cost_per_connection(df):
    """03 — Box-plot of capital cost per connection."""
    fig, ax = plt.subplots(figsize=(12, 7))
    data, labels, colors = [], [], []
    for col, label in [("grid_cost_per_conn", "Grid"),
                       ("minigrid_cost_per_conn", "Mini-grid"),
                       ("shs_cost_per_conn", "SHS")]:
        vals = df[col].dropna()
        vals = vals[vals < np.inf]
        if len(vals):
            data.append(vals)
            labels.append(label)
            colors.append(TECH_COLORS.get(label, "#95a5a6"))

    bp = ax.boxplot(data, labels=labels, patch_artist=True, showmeans=True)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    ax.set_ylabel("Cost per Connection (USD)", fontsize=12, fontweight="bold")
    ax.set_title("Capital Cost per Connection by Technology",
                 fontsize=14, fontweight="bold", pad=20)
    ax.grid(True, alpha=0.3)
    save(fig, "03_cost_per_connection.png")


def plot_technology_by_distance(df):
    """04 — Scatter: technology selection vs distance & population."""
    fig, ax = plt.subplots(figsize=(12, 7))
    for tech in df["least_cost_technology"].unique():
        sub = df[df["least_cost_technology"] == tech]
        ax.scatter(sub["distance_to_grid_km"], sub["population"],
                   label=tech, alpha=0.6, s=80,
                   color=TECH_COLORS.get(tech, "#95a5a6"))
    ax.set_xlabel("Distance to Grid (km)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Population", fontsize=12, fontweight="bold")
    ax.set_title("Technology Selection by Settlement Location",
                 fontsize=14, fontweight="bold", pad=20)
    ax.set_yscale("log")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    save(fig, "04_technology_by_distance.png")


def plot_population_vs_lcoe(df):
    """05 — Scatter: population vs least-cost LCOE, coloured by tech."""
    fig, ax = plt.subplots(figsize=(12, 7))
    for tech in df["least_cost_technology"].unique():
        sub = df[df["least_cost_technology"] == tech]
        ax.scatter(sub["population"], sub["least_cost_lcoe"],
                   label=tech, alpha=0.6, s=80,
                   color=TECH_COLORS.get(tech, "#95a5a6"))
    ax.set_xlabel("Population", fontsize=12, fontweight="bold")
    ax.set_ylabel("Least-Cost LCOE (USD / MWh)", fontsize=12, fontweight="bold")
    ax.set_title("Electrification Cost vs Settlement Size",
                 fontsize=14, fontweight="bold", pad=20)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    save(fig, "05_population_vs_lcoe.png")


def plot_demand_tier_distribution(df):
    """06 — Bar chart of demand-tier distribution."""
    fig, ax = plt.subplots(figsize=(10, 6))
    tier_counts = df["demand_tier"].value_counts().sort_index()
    bars = ax.bar(tier_counts.index, tier_counts.values,
                  color=["#f1c40f", "#e67e22", "#e74c3c", "#8e44ad", "#2c3e50"])
    for bar, val in zip(bars, tier_counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                str(val), ha="center", va="bottom", fontweight="bold")
    ax.set_xlabel("Demand Tier (World Bank MTF)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Number of Settlements", fontsize=12, fontweight="bold")
    ax.set_title("Settlement Distribution by Demand Tier",
                 fontsize=14, fontweight="bold", pad=20)
    ax.grid(True, alpha=0.3, axis="y")
    save(fig, "06_demand_tier_distribution.png")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def plot_sensitivity_lines(sens):
    """10 — 5-panel line chart: tech mix vs each parameter value."""
    params = sens["parameter"].unique()
    fig, axes = plt.subplots(1, len(params), figsize=(18, 6), sharey=False)
    if len(params) == 1:
        axes = [axes]

    COLORS = {"Grid": TECH_COLORS["Grid"],
              "Mini-grid": TECH_COLORS["Mini-grid"],
              "SHS": TECH_COLORS["SHS"]}

    for ax, param in zip(axes, params):
        sub   = sens[sens["parameter"] == param].copy()
        meta  = SENSITIVITY_PARAMS[param]
        xlabs = sub["value_label"].tolist()
        x     = range(len(xlabs))

        ax.plot(x, sub["grid_pct"],     marker="o", color=COLORS["Grid"],
                linewidth=2, label="Grid")
        ax.plot(x, sub["minigrid_pct"], marker="s", color=COLORS["Mini-grid"],
                linewidth=2, label="Mini-grid")
        ax.plot(x, sub["shs_pct"],      marker="^", color=COLORS["SHS"],
                linewidth=2, label="SHS")

        # Shade base-case column
        base_idx = sub[sub["is_base"]].index
        if len(base_idx):
            bi = sub.index.get_loc(base_idx[0])
            ax.axvspan(bi - 0.3, bi + 0.3, color="grey", alpha=0.12)

        ax.set_xticks(list(x))
        ax.set_xticklabels(xlabs, rotation=30, ha="right", fontsize=8)
        ax.set_title(meta["label"], fontsize=10, fontweight="bold")
        ax.set_ylabel("% of Settlements", fontsize=9)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("Sensitivity Analysis — Technology Mix vs Parameter Value",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "10_sensitivity_lines.png", dpi=300, bbox_inches="tight")
    plt.close()
    log.info("  Saved: 10_sensitivity_lines.png")


def plot_tornado(sens):
    """11 — Tornado chart: parameter impact on % SHS share (most sensitive metric)."""
    rows = []
    for param in sens["parameter"].unique():
        sub  = sens[sens["parameter"] == param]
        base = sub[sub["is_base"]]["shs_pct"].values[0]
        lo   = sub["shs_pct"].min()
        hi   = sub["shs_pct"].max()
        rows.append({
            "label": SENSITIVITY_PARAMS[param]["label"],
            "base":  base,
            "swing": hi - lo,
            "lo":    lo - base,
            "hi":    hi - base,
        })
    tdf = pd.DataFrame(rows).sort_values("swing", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    y = range(len(tdf))
    bars_lo = ax.barh(list(y), tdf["lo"], left=tdf["base"],
                      color=TECH_COLORS["Grid"], alpha=0.8, label="Decrease in SHS %")
    bars_hi = ax.barh(list(y), tdf["hi"], left=tdf["base"],
                      color=TECH_COLORS["SHS"], alpha=0.8, label="Increase in SHS %")
    ax.axvline(tdf["base"].iloc[0], color="black", linewidth=1.5,
               linestyle="--", label="Base case")

    ax.set_yticks(list(y))
    ax.set_yticklabels(tdf["label"], fontsize=11)
    ax.set_xlabel("SHS Share (% of settlements)", fontsize=11, fontweight="bold")
    ax.set_title("Tornado Chart — Parameter Impact on SHS Share",
                 fontsize=13, fontweight="bold", pad=15)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="x")

    # Annotate swing width
    for i, row in enumerate(tdf.itertuples()):
        ax.text(max(row.base + row.hi, row.base + row.lo) + 0.5, i,
                f"±{row.swing/2:.1f}pp", va="center", fontsize=9, color="dimgray")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "11_tornado_chart.png", dpi=300, bbox_inches="tight")
    plt.close()
    log.info("  Saved: 11_tornado_chart.png")


def plot_sensitivity_lcoe(sens):
    """12 — Mean LCOE change across parameter values."""
    params = sens["parameter"].unique()
    fig, axes = plt.subplots(1, len(params), figsize=(18, 5), sharey=False)
    if len(params) == 1:
        axes = [axes]

    for ax, param in zip(axes, params):
        sub   = sens[sens["parameter"] == param]
        meta  = SENSITIVITY_PARAMS[param]
        xlabs = sub["value_label"].tolist()
        x     = range(len(xlabs))

        ax.plot(x, sub["mean_lcoe"], marker="D", color=NAVY, linewidth=2)
        ax.fill_between(x, sub["mean_lcoe"].min(), sub["mean_lcoe"],
                        alpha=0.12, color=NAVY)

        base_idx = sub[sub["is_base"]].index
        if len(base_idx):
            bi = sub.index.get_loc(base_idx[0])
            ax.axvspan(bi - 0.3, bi + 0.3, color="grey", alpha=0.12)

        ax.set_xticks(list(x))
        ax.set_xticklabels(xlabs, rotation=30, ha="right", fontsize=8)
        ax.set_title(meta["label"], fontsize=10, fontweight="bold")
        ax.set_ylabel("Mean LCOE (USD/MWh)", fontsize=9)
        ax.grid(True, alpha=0.3)

    NAVY_RGB = (30/255, 39/255, 97/255)
    fig.suptitle("Sensitivity Analysis — Impact on Mean LCOE",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "12_sensitivity_lcoe.png", dpi=300, bbox_inches="tight")
    plt.close()
    log.info("  Saved: 12_sensitivity_lcoe.png")


NAVY = "#1E2761"


def main():
    log.info("=" * 70)
    log.info("BENIN ELECTRIFICATION — PIPELINE START")
    log.info("=" * 70)

    # ── Phase 1: Run the core analysis ─────────────────────────────
    log.info("\nPHASE 1: Running demand + cost analysis ...")
    df = run_analysis(str(DATA_FILE), str(OUTPUT_DIR))

    # ── Phase 2: Summary statistics ────────────────────────────────
    log.info("\nPHASE 2: Summary statistics")

    log.info(f"  Settlements analysed : {len(df)}")
    log.info(f"  Total population     : {df['population'].sum():,}")
    log.info(f"  Median dist to grid  : {df['distance_to_grid_km'].median():.1f} km")
    log.info(f"  Health facilities    : {df['has_health'].sum()}")
    log.info(f"  Education facilities : {df['has_education'].sum()}")

    tech = df["least_cost_technology"].value_counts()
    log.info("\n  Technology distribution:")
    for t, n in tech.items():
        pct = 100 * n / len(df)
        log.info(f"    {t:12s}  {n:4d}  ({pct:.1f} %)")

    log.info("\n  LCOE by technology (USD/MWh, viable only):")
    for col, label in [("grid_lcoe", "Grid"),
                       ("minigrid_lcoe", "Mini-grid"),
                       ("shs_lcoe", "SHS")]:
        vals = df[col].dropna()
        if len(vals):
            log.info(f"    {label:12s}  median {vals.median():>8,.0f}  "
                     f"mean {vals.mean():>8,.0f}")

    log.info("\n  Cost per connection (USD, viable only):")
    for col, label in [("grid_cost_per_conn", "Grid"),
                       ("minigrid_cost_per_conn", "Mini-grid"),
                       ("shs_cost_per_conn", "SHS")]:
        vals = df[col].dropna()
        if len(vals):
            log.info(f"    {label:12s}  median {vals.median():>8,.0f}  "
                     f"mean {vals.mean():>8,.0f}")

    # ── Phase 3: Visualisations ────────────────────────────────────
    log.info("\nPHASE 3: Generating visualisations ...")
    plot_technology_distribution(df)
    plot_lcoe_comparison(df)
    plot_cost_per_connection(df)
    plot_technology_by_distance(df)
    plot_population_vs_lcoe(df)
    plot_demand_tier_distribution(df)

    # ── Phase 4: Priority settlements ──────────────────────────────
    log.info("\nPHASE 4: Top-priority settlements")
    top = df.nlargest(10, "priority_score")
    for _, r in top.iterrows():
        log.info(f"  {r['name']:20s}  pop {r['population']:>7,}  "
                 f"{r['least_cost_technology']:10s}  "
                 f"LCOE ${r['least_cost_lcoe']:>7,.0f}  "
                 f"priority {r['priority_score']:.3f}")

    # ── Phase 5: Sensitivity analysis ─────────────────────────────
    log.info("\nPHASE 5: Running sensitivity analysis (1,000 settlements x 25 runs)...")
    sens = run_sensitivity(str(DATA_FILE), str(OUTPUT_DIR), sample_n=1000)
    log.info("\nPHASE 5b: Generating sensitivity charts ...")
    plot_sensitivity_lines(sens)
    plot_tornado(sens)
    plot_sensitivity_lcoe(sens)

    # ── Done ───────────────────────────────────────────────────────
    log.info("\n" + "=" * 70)
    log.info("PIPELINE COMPLETE")
    log.info("=" * 70)
    log.info(f"Outputs in: {OUTPUT_DIR.absolute()}")
    log.info("  - benin_electrification_results.csv")
    log.info("  - top_20_priority_settlements.csv")
    log.info("  - sensitivity_results.csv")
    log.info("  - 01-12_*.png")

    return df


if __name__ == "__main__":
    results = main()
