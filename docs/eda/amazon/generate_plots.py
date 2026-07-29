"""Regenerate the Amazon Fashion dataset ETL/EDA plots embedded in the top-level README.

Run from anywhere:  python3 docs/eda/amazon/generate_plots.py
Reads:  amazon/canonical/{structured,texts}.csv
Writes: docs/eda/amazon/*.png
"""
import os
import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = HERE

# --- palette (dataviz skill, light mode, validated categorical order) ---
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": BASELINE, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "font.size": 11,
})


def style_ax(ax, hide_spines=("top", "right", "left")):
    for s in hide_spines:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(length=0)
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)


def parse_price(s):
    if pd.isna(s):
        return None
    s = str(s).replace("$", "").replace(",", "").strip()
    m = re.match(r"^([\d.]+)", s)
    return float(m.group(1)) if m else None


def main():
    struct = pd.read_csv(os.path.join(REPO_ROOT, "amazon/canonical/structured.csv"))
    texts = pd.read_csv(os.path.join(REPO_ROOT, "amazon/canonical/texts.csv"))
    struct["price_num"] = struct["price"].apply(parse_price)

    # 1. Structured field completeness
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=200)
    cols = ["title", "brand", "price", "description", "features", "sales_rank"]
    completeness = [(1 - struct[c].isna().mean()) * 100 for c in cols]
    order = np.argsort(completeness)
    cols_sorted = [cols[i] for i in order]
    comp_sorted = [completeness[i] for i in order]
    ax.barh(cols_sorted, comp_sorted, color=BLUE, zorder=3, height=0.6)
    style_ax(ax, hide_spines=("top", "right"))
    ax.grid(axis="x", zorder=0)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Rows populated (%)")
    ax.set_title(f"Amazon Fashion structured fields: completeness (n = {len(struct):,} products)",
                 loc="left", fontsize=11.5, color=INK, fontweight="bold")
    for i, v in enumerate(comp_sorted):
        ax.text(v + 1.5, i, f"{v:.1f}%", va="center", fontsize=9, color=INK2)
    fig.tight_layout()
    fig.savefig(f"{OUT}/01_field_completeness.png", facecolor=SURFACE)
    plt.close(fig)

    # 2. Price distribution (clipped, among the populated subset)
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=200)
    price = struct["price_num"].dropna()
    clipped = price[price <= 150]
    pct_shown = len(clipped) / len(price) * 100
    ax.hist(clipped, bins=40, color=BLUE, zorder=3, edgecolor=SURFACE, linewidth=0.3)
    style_ax(ax)
    ax.set_xlabel("Price (USD)")
    ax.set_ylabel("Products (count)")
    ax.set_title(f"Price distribution, clipped to $150 (of {len(price):,} priced products)",
                 loc="left", fontsize=11.5, color=INK, fontweight="bold")
    ax.text(0.98, 0.95,
            f"price populated for only\n{len(price)/len(struct)*100:.1f}% of products;\n{pct_shown:.1f}% of those are ≤ $150",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color=INK2)
    fig.tight_layout()
    fig.savefig(f"{OUT}/02_price_distribution.png", facecolor=SURFACE)
    plt.close(fig)

    # 3. Rating distribution + verified purchase share
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), dpi=200)
    ax = axes[0]
    rating_counts = texts["rating"].value_counts().sort_index()
    ax.bar(rating_counts.index.astype(int).astype(str), rating_counts.values, color=BLUE, zorder=3, width=0.6)
    style_ax(ax)
    ax.set_xlabel("Star rating")
    ax.set_ylabel("Reviews (count)")
    ax.set_title("Star rating distribution", loc="left", fontsize=11.5, color=INK, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x/1000)}k" if x >= 1000 else int(x)))

    ax2 = axes[1]
    verified_counts = texts["verified"].value_counts()
    labels = ["Verified", "Unverified"]
    values = [verified_counts.get(True, 0), verified_counts.get(False, 0)]
    bars = ax2.bar(labels, values, color=[AQUA, ORANGE], zorder=3, width=0.55)
    style_ax(ax2)
    ax2.set_ylabel("Reviews (count)")
    ax2.set_title("Verified purchase share", loc="left", fontsize=11.5, color=INK, fontweight="bold")
    for b, v in zip(bars, values):
        ax2.text(b.get_x() + b.get_width() / 2, v + 8000, f"{v/len(texts)*100:.0f}%", ha="center", fontsize=9, color=INK2)
    fig.suptitle(f"Amazon Fashion review corpus (n = {len(texts):,})", x=0.02, ha="left",
                 fontsize=12, color=INK, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUT}/03_review_ratings_verified.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    # 4. Review length distribution
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=200)
    lens = texts["review_text"].astype(str).str.len()
    clipped_len = lens[lens <= 1000]
    ax.hist(clipped_len, bins=40, color=BLUE, zorder=3, edgecolor=SURFACE, linewidth=0.3)
    style_ax(ax)
    ax.set_xlabel("Review length (characters)")
    ax.set_ylabel("Reviews (count)")
    ax.set_title(f"Review length distribution, clipped to 1,000 chars ({(lens<=1000).mean()*100:.0f}% of reviews)",
                 loc="left", fontsize=11.5, color=INK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{OUT}/04_review_length.png", facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    main()
