import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    from scripts.extract_dataset import group_by_last_author
    from scripts.iclr_analysis import compute_text_embeddings_SBERT, download_autotokenizer
    import matplotlib.pyplot as plt
    import numpy as np

    return compute_text_embeddings_SBERT, group_by_last_author, mo, np, pd, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Importing the ICLR OpenReview Dataset
    """)
    return


@app.cell
def _(pd):
    iclr2026v1 = pd.read_parquet("data/iclr26v1.parquet")
    iclr2026v1.head()
    return (iclr2026v1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Group Papers by Principal Investigator (Last Listed Author)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here we are assuming that the last author listed in `authors` is the Principal Investigators (PIs). We want to analyze the PIs because they have more publications than a new researcher.
    """)
    return


@app.cell
def _(group_by_last_author, iclr2026v1):
    last_author_df = group_by_last_author(iclr2026v1)
    last_author_df.sort_values("num_papers", ascending=False)
    return (last_author_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Initial Analysis
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Visualization: Complementary Cumulative Distribution Function (CCDF)
    """)
    return


@app.cell
def _(last_author_df, np, plt):
    paper_counts = last_author_df["num_papers"].values
    n_total = len(paper_counts)

    k_values = np.arange(1, 26)
    retention_pct = np.array([(paper_counts >= k).mean() * 100 for k in k_values])

    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.step(k_values, retention_pct, where="post", color="crimson", linewidth=2.5)
    ax1.set_xlabel("Minimum Papers Threshold (k)", fontsize=11)
    ax1.set_ylabel("% of Authors Retained", color="crimson", fontsize=11)
    ax1.tick_params(axis="y", labelcolor="crimson")
    ax1.set_xlim(1, 25)
    ax1.set_ylim(0, 105)
    ax1.grid(True, linestyle="--", alpha=0.3)


    ax2 = ax1.twinx()
    ax2.set_ylabel("Number of Authors Retained", color="navy", fontsize=11)
    ax2.set_ylim(0, n_total * 1.05)
    ax2.tick_params(axis="y", labelcolor="navy")

    for k in [2, 3, 6, 10]:
        pct = (paper_counts >= k).mean() * 100
        cnt = (paper_counts >= k).sum()
        ax1.plot(k, pct, marker="o", markersize=6, color="black", zorder=5)
        ax1.annotate(
            f"k={k}: {pct:.1f}%\n({cnt:,} authors)",
            (k, pct),
            textcoords="offset points",
            xytext=(10, -18 if k in [2, 3] else 10),
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="gray", alpha=0.85),
            arrowprops=dict(arrowstyle="->", lw=0.8, color="black")
        )

    plt.title("Complementary Cumulative Distribution Function (CCDF)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For this inital analysis, we have decided to look at last listed authors who are part of 5 papers or more. This is because we want to study researchers who published more than 1 paper. These are the authors who decided to stay within a field and to change fields.

    However, there are researchers who do not collaborate with other people. This is included in our dataset. Are these independent researchers considered PIs? Let's look at the number of independent researchers in the dataset.
    """)
    return


@app.cell
def _(last_author_df):
    solo_authors = last_author_df[last_author_df["collaborators"].apply(lambda collabs: all(len(c) == 0 for c in collabs))]
    solo_authors.sort_values("num_papers", ascending=False)
    return (solo_authors,)


@app.cell
def _(solo_authors):
    print(f"Authors with only solo papers: {len(solo_authors)}")
    print(f"Average number of publications for solo-only authors: {solo_authors['num_papers'].mean():.2f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here we will eliminate papers who list "Anonymous" as the author.
    """)
    return


@app.cell
def _(last_author_df):
    threshold = 5
    df = last_author_df[(last_author_df["num_papers"] >= threshold) & (last_author_df["last_author"] != "Anonymous")].reset_index(drop=True)
    df
    return df, threshold


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Distribution of Papers per Author
    """)
    return


@app.cell
def _(df, plt):
    paper_freq = df["num_papers"].value_counts().sort_index()

    plt.figure(figsize=(10, 5))
    plt.vlines(
        x=paper_freq.index,
        ymin=0,
        ymax=paper_freq.values,
        color="dodgerblue",
        linestyle="--",
        alpha=0.4,
        linewidth=1,
    )
    plt.scatter(
        paper_freq.index,
        paper_freq.values,
        color="dodgerblue",
        edgecolor="blue",
        alpha=0.85,
        s=40,
        zorder=3,
    )
    plt.xticks(range(0, 125, 5))
    plt.xlim(left=-1, right=122)
    plt.title("Number of Authors vs. Number of Papers (No log)")
    plt.xlabel("Number of Papers")
    plt.ylabel("Number of Authors")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.gca()
    return (paper_freq,)


@app.cell
def _(paper_freq, plt, threshold):
    plt.figure(figsize=(10, 5))

    plt.scatter(
        paper_freq.index,
        paper_freq.values,
        color="dodgerblue",
        edgecolor="blue",
        alpha=0.85,
        s=40,
        zorder=3,
    )
    plt.xscale("log")
    plt.yscale("log")

    plt.xlim(left=threshold * 0.85, right=paper_freq.index.max() * 1.2)
    plt.ylim(bottom=0.8, top=paper_freq.values.max() * 1.5)

    x_ticks = [5, 10, 15, 20, 30, 50, 75, 100]
    plt.xticks(x_ticks, x_ticks)

    y_ticks = [1, 2, 5, 10, 20, 50, 100, 200, 350]
    plt.yticks(y_ticks, y_ticks)

    plt.title("Number of Authors vs. Number of Papers (Log-Log Scale)")
    plt.xlabel("Number of Papers (log scale)")
    plt.ylabel("Number of Authors (log scale)")
    plt.tight_layout()
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Paper Representation: Recreating the Embeddings
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The authors of the original paper used multiple models to convert the abstracts into representations. One approach was to use the SBERT. The results from the paper shows that SBERT had a higher performance to the other methods. The authors do not provide their weights, but fortunately they provide the code to recreate their embeddings.
    """)
    return


@app.cell
def _(compute_text_embeddings_SBERT, iclr2026v1):
    autotokenizer_path = "/scratch/network/al5098/COS424/autotokenizer"
    output_dir = "/scratch/network/al5098/COS424/"

    compute_text_embeddings_SBERT(iclr2026v1, local_path=autotokenizer_path, output_dir=output_dir)
    return


if __name__ == "__main__":
    app.run()
