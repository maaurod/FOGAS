
# ══════════════════════════════════════════════════════════════════
#  Dataset-Variation Grid Search — Results Plotter  (40-grid)
#
#  CSV inputs  (DATA_DIR):
#    grid_search_dataset_40grid_A.csv  – Family A (Manual Augmentation)
#    grid_search_dataset_40grid_B.csv  – Family B (Epsilon Variation)
#    grid_search_dataset_40grid_C.csv  – Family C (Random-Start Coverage)
#
#  For each family one figure per metric is produced:
#    Convergence Rate | Coverage Fraction | Final Reward | Q-Optimality Gap
#  Fixed x-axis per family:
#    A → n_uniform   |   B → epsilon   |   C → p_rand
# ══════════════════════════════════════════════════════════════════
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ── Config ────────────────────────────────────────────────────────
FOGAS_ROOT = Path("/shared/home/mauro.diaz/work/FOGAS")
DATA_DIR   = FOGAS_ROOT / "datasets" / "grids"
OUTPUT_DIR = FOGAS_ROOT / "testing_vectorized" / "plots"
SAVE_FIGS  = False      # True → write PNGs; False → show inline
DPI        = 150

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="viridis")
plt.rcParams["figure.dpi"] = 100


# ── Helpers ───────────────────────────────────────────────────────
def load(name: str) -> pd.DataFrame | None:
    p = DATA_DIR / name
    if not p.exists():
        print(f"⚠️  Not found: {p}")
        return None
    df = pd.read_csv(p)
    print(f"✅ {p.name}: {len(df)} rows  |  columns: {list(df.columns)}")
    return df


def show_or_save(fig, name: str):
    plt.tight_layout()
    if SAVE_FIGS:
        out = OUTPUT_DIR / f"{name}.png"
        fig.savefig(out, dpi=DPI, bbox_inches="tight")
        print(f"📸 Saved: {out}")
    else:
        plt.show()
    plt.close(fig)


def plot_family(df, x_col, y_vars, family_title, xlabel, file_prefix,
                color="steelblue", marker="o"):
    """
    One figure per y-metric.  Each figure has a single axis with a seaborn
    lineplot (mean ± SEM band) of y_var vs x_col.

    Parameters
    ----------
    df          : DataFrame for this family
    x_col       : column to use as x-axis
    y_vars      : list of (col_name, display_label, invert_bool)
    family_title: string used in suptitle
    xlabel      : x-axis label
    file_prefix : prefix for saved PNG names
    color       : line / marker colour
    marker      : marker style
    """
    print(f"\n📊 Plotting {family_title} …")
    for col, ylabel, invert in y_vars:
        if col not in df.columns:
            print(f"   ⚠️  Column '{col}' not found – skipping '{ylabel}'.")
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        fig.suptitle(
            f"{family_title}\n{ylabel}  vs  {xlabel}",
            fontsize=15, fontweight="bold", y=1.02,
        )

        sns.lineplot(
            data=df, x=x_col, y=col,
            marker=marker, markersize=8, linewidth=2,
            color=color, err_style="band",
            ax=ax,
        )

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title("")          # title already in suptitle
        ax.grid(True, linestyle="--", alpha=0.5)

        if invert:
            ax.invert_yaxis()

        show_or_save(fig, f"{file_prefix}_{col}")


# ── Metrics (shared across all three families) ────────────────────
# (column_name, display_label, invert_y_axis?)
Y_VARS = [
    ("convergence",  "Convergence Rate",    False),
    ("coverage",     "Coverage Fraction",   False),
    ("final_reward", "Final Reward",         False),
    ("q_gap",        "Q-Optimality Gap ↓",  True),
]


# ══════════════════════════════════════════════════════════════════
#  Family A – Manual Augmentation
#  x-axis: n_uniform  (samples per unvisited (s,a) pair)
# ══════════════════════════════════════════════════════════════════
df_A = load("grid_search_dataset_40grid_A.csv")

if df_A is not None:
    plot_family(
        df=df_A,
        x_col="n_uniform",
        y_vars=Y_VARS,
        family_title="Family A – Manual Augmentation",
        xlabel="Samples per Unvisited (s,a) Pair  (n_uniform)",
        file_prefix="plot_A",
        color="steelblue",
        marker="o",
    )


# ══════════════════════════════════════════════════════════════════
#  Family B – Epsilon Variation
#  x-axis: epsilon
# ══════════════════════════════════════════════════════════════════
df_B = load("grid_search_dataset_40grid_B.csv")

if df_B is not None:
    plot_family(
        df=df_B,
        x_col="epsilon",
        y_vars=Y_VARS,
        family_title="Family B – Epsilon Variation",
        xlabel="Epsilon (ε)",
        file_prefix="plot_B",
        color="darkorange",
        marker="s",
    )


# ══════════════════════════════════════════════════════════════════
#  Family C – Random-Start Policy Coverage
#  x-axis: p_rand
# ══════════════════════════════════════════════════════════════════
df_C = load("grid_search_dataset_40grid_C.csv")

if df_C is not None:
    plot_family(
        df=df_C,
        x_col="p_rand",
        y_vars=Y_VARS,
        family_title="Family C – Random-Start Coverage",
        xlabel="Fraction of Random-Start Samples (p_rand)",
        file_prefix="plot_C",
        color="seagreen",
        marker="^",
    )


print("\n✅ Done.")
