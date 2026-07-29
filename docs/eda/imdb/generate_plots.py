"""Regenerate the IMDb dataset ETL/EDA plots embedded in the top-level README.

Run from anywhere:  python3 docs/eda/imdb/generate_plots.py
Reads:  imdb/data/canonical/{imdb_structured,imdb_reviews}.csv
Writes: docs/eda/imdb/*.png
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = HERE

# --- palette (dataviz skill, light mode, validated categorical order) ---
BLUE, ORANGE = "#2a78d6", "#eb6834"
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


def main():
    struct = pd.read_csv(os.path.join(REPO_ROOT, "imdb/data/canonical/imdb_structured.csv"))
    reviews = pd.read_csv(os.path.join(REPO_ROOT, "imdb/data/canonical/imdb_reviews.csv"))

    # 1. Movies per decade
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=200)
    decade = (struct["year"] // 10 * 10)
    decade = decade[(decade >= 1890) & (decade <= 2030)]
    counts = decade.value_counts().sort_index()
    ax.bar(counts.index.astype(str), counts.values, width=0.7, color=BLUE, zorder=3)
    style_ax(ax)
    ax.set_ylabel("Movies (count)")
    ax.set_title(f"IMDb structured catalog: movies by decade (n = {len(struct):,})",
                 loc="left", fontsize=12, color=INK, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x/1000)}k" if x >= 1000 else int(x)))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(f"{OUT}/01_movies_by_decade.png", facecolor=SURFACE)
    plt.close(fig)

    # 2. Runtime distribution (clipped)
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=200)
    rt = struct["runtime"]
    clipped = rt[(rt > 0) & (rt <= 240)]
    pct_outlier = 1 - len(clipped) / len(rt)
    ax.hist(clipped, bins=48, color=BLUE, zorder=3, edgecolor=SURFACE, linewidth=0.3)
    style_ax(ax)
    ax.set_xlabel("Runtime (minutes)")
    ax.set_ylabel("Movies (count)")
    ax.set_title("Runtime distribution, clipped to 0-240 min", loc="left", fontsize=12, color=INK, fontweight="bold")
    ax.text(0.98, 0.95, f"{pct_outlier*100:.1f}% of rows fall outside\nthis range (data artifacts,\ne.g. runtime = 51,420 min)",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color=INK2)
    fig.tight_layout()
    fig.savefig(f"{OUT}/02_runtime_distribution.png", facecolor=SURFACE)
    plt.close(fig)

    # 3. Top genre combinations
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=200)
    top_genres = struct["genres"].value_counts().head(10).sort_values()
    ax.barh(top_genres.index, top_genres.values, color=BLUE, zorder=3, height=0.65)
    style_ax(ax, hide_spines=("top", "right"))
    ax.grid(axis="x", zorder=0)
    ax.set_xlabel("Movies (count)")
    ax.set_title("Top 10 genre combinations", loc="left", fontsize=12, color=INK, fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x/1000)}k" if x >= 1000 else int(x)))
    fig.tight_layout()
    fig.savefig(f"{OUT}/03_top_genres.png", facecolor=SURFACE)
    plt.close(fig)

    # 4. Review corpus: sentiment balance + length distribution
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), dpi=200)
    ax = axes[0]
    label_counts = reviews["label"].value_counts().sort_index()
    bars = ax.bar(["Negative (0)", "Positive (1)"], label_counts.values, color=[ORANGE, BLUE], zorder=3, width=0.55)
    style_ax(ax)
    ax.set_ylabel("Reviews (count)")
    ax.set_title("Sentiment label balance", loc="left", fontsize=11.5, color=INK, fontweight="bold")
    for b, v in zip(bars, label_counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 200, f"{v:,}", ha="center", fontsize=9, color=INK2)

    ax2 = axes[1]
    review_len = reviews["review"].str.len()
    clipped_len = review_len[review_len <= 4000]
    ax2.hist(clipped_len, bins=40, color=BLUE, zorder=3, edgecolor=SURFACE, linewidth=0.3)
    style_ax(ax2)
    ax2.set_xlabel("Review length (characters)")
    ax2.set_ylabel("Reviews (count)")
    ax2.set_title("Review length distribution", loc="left", fontsize=11.5, color=INK, fontweight="bold")
    fig.suptitle(f"IMDb-ID/Stanford review split (n = {len(reviews):,})", x=0.02, ha="left",
                 fontsize=12, color=INK, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUT}/04_review_corpus.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
