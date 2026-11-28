"""
Paper-level specialization measures for the PI of each paper.

Team decisions this module follows:
    * Only papers from 2021 on are used: earlier papers have no OpenReview author ids,
      so authors can only be matched by name and namesakes get merged.
      Every author (the PI included) is identified by their OpenReview id.
    * A resubmitted paper is kept only once: the accepted version if there is one,
      otherwise the latest version ("accepted_or_latest"). The alternative is to drop
      every paper that belongs to a chain of resubmissions ("drop_all").

The PI of a paper is its last listed author. For a paper by PI `a` submitted in
year `t`, the PI's history H(a, t) is every paper in which `a` appears (in any
author position) from years strictly before `t`. Papers from the same year are
never part of the history.

Measures computed for each PI paper:
    pivot        1 - cos(paper, mean of all prior papers)          (Hill et al., 2025)
    pivot_nn     1 - cos(paper, most similar prior paper)
    pivot_recent 1 - cos(paper, exponentially weighted prior mean), half-life of 2 years
    vendi        Vendi score of m prior papers: how many distinct directions they span
                 (1 = all identical, m = all orthogonal), averaged over random subsets
                 of size m so that every PI is measured on the same number of papers.

All embeddings are L2-normalised first, so a dot product is a cosine similarity.
"""

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_master(
    path: str = "../embeddings/master.parquet",
    min_year: int = 2021,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Loads the master dataframe made by data/generate.py.

    Output:
        papers: one row per paper from `min_year` on. The index is the paper's row in
                the embeddings (it is not reset), so embeddings[paper_idx] always works.
        embeddings: (36070, 768) array with unit-length rows, for every paper.
    """
    master = pd.read_parquet(path)
    embeddings = np.stack(master["embedding"]).astype(np.float64)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    papers = master.drop(columns="embedding")
    papers = papers[papers["year"] >= min_year]
    return papers, embeddings


def split_list(raw: str) -> list[str]:
    """Splits a comma-separated string (author names or ids) into trimmed items, order kept."""
    if not isinstance(raw, str):
        return []
    items = [item.strip() for item in raw.split(",")]
    return [item for item in items if item]


# ---------------------------------------------------------------------------
# Resubmissions
# ---------------------------------------------------------------------------
def find_resubmission_chains(
    papers: pd.DataFrame,
    embeddings: np.ndarray,
    similarity_threshold: float = 0.90,
) -> pd.Series:
    """
    Groups papers that are versions of the same submission.

    Two papers are linked when they share an author id, come from different years and
    have a cosine similarity of at least `similarity_threshold`. Linked pairs are
    joined into chains (A -> B -> C).

    Output:
        chain: Series aligned with `papers`; papers in the same chain share a value.
               A paper that is not part of any chain has a chain of its own.
    """
    papers_by_author = {}
    for paper_idx, raw_ids in zip(papers.index, papers["author_ids"]):
        for author_id in set(split_list(raw_ids)):
            papers_by_author.setdefault(author_id, []).append(paper_idx)

    year_of = papers["year"]
    linked_pairs = set()

    for indices in papers_by_author.values():
        if len(indices) < 2:
            continue
        indices = np.array(indices)
        similarity = embeddings[indices] @ embeddings[indices].T
        rows, cols = np.where(np.triu(similarity >= similarity_threshold, k=1))
        for r, c in zip(rows, cols):
            if year_of[indices[r]] != year_of[indices[c]]:
                linked_pairs.add((indices[r], indices[c]))

    n_rows = len(embeddings)   # graph over embedding rows, since the index is not reset
    if linked_pairs:
        pairs = np.array(sorted(linked_pairs))
        graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n_rows, n_rows))
        _, chain_of_row = connected_components(graph, directed=False)
    else:
        chain_of_row = np.arange(n_rows)

    return pd.Series(chain_of_row[papers.index], index=papers.index, name="chain")


def resubmissions_to_drop(papers: pd.DataFrame, chain: pd.Series, rule: str = "accepted_or_latest") -> set:
    """
    Chooses which papers in chains of resubmissions to leave out.

    rule = "accepted_or_latest": keep the accepted version if there is one, otherwise the
                                 latest one (the team's rule).
    rule = "drop_all":           leave out every paper that is part of a chain.
    """
    chain_size = chain.map(chain.value_counts())
    in_chain = papers[chain_size > 1].assign(chain=chain)

    if rule == "accepted_or_latest":
        kept = in_chain.sort_values(["accepted", "year"]).groupby("chain").tail(1).index
    elif rule == "drop_all":
        kept = []
    else:
        raise ValueError(f"Unknown rule: {rule}")

    return set(in_chain.index) - set(kept)


# ---------------------------------------------------------------------------
# Author histories
# ---------------------------------------------------------------------------
def build_author_papers(papers: pd.DataFrame, drop: set) -> pd.DataFrame:
    """
    One row per (paper, author id), with whether the author is the PI (last author).
    Papers in `drop` are left out, both as observations and from every history.
    """
    records = []
    for paper_idx, row in papers.iterrows():
        if paper_idx in drop:
            continue
        author_ids = list(dict.fromkeys(split_list(row["author_ids"])))
        pi_id = split_list(row["author_ids"])[-1]
        for author_id in author_ids:
            records.append({
                "paper_idx": paper_idx,
                "year": row["year"],
                "author_id": author_id,
                "is_pi": author_id == pi_id,
            })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Measures
# ---------------------------------------------------------------------------
def cosine_distance_to_mean(paper: np.ndarray, prior: np.ndarray, weights: np.ndarray | None = None) -> float:
    """1 - cosine similarity between a paper and the (weighted) mean of prior papers."""
    centroid = np.average(prior, axis=0, weights=weights)
    centroid = centroid / np.linalg.norm(centroid)
    return 1.0 - paper @ centroid


def vendi_score(vectors: np.ndarray) -> float:
    """
    Vendi score (Friedman & Dieng, 2023) with cosine similarity.

    K is the n x n cosine-similarity matrix (ones on the diagonal). The eigenvalues of
    K / n sum to one; the score is the exponential of their Shannon entropy, i.e. the
    effective number of distinct directions among the n vectors (between 1 and n).
    """
    n = len(vectors)
    similarity = vectors @ vectors.T
    eigenvalues = np.linalg.eigvalsh(similarity / n)
    eigenvalues = eigenvalues[eigenvalues > 1e-12]
    entropy = -np.sum(eigenvalues * np.log(eigenvalues))
    return float(np.exp(entropy))


def rarefied_vendi(prior: np.ndarray, m: int, n_subsets: int, rng: np.random.Generator) -> float:
    """
    Vendi score on exactly m papers, averaged over random subsets of the history.
    Using a fixed m makes the score comparable between PIs with short and long histories.
    """
    if len(prior) < m:
        return np.nan
    if len(prior) == m:
        return vendi_score(prior)
    scores = [vendi_score(prior[rng.choice(len(prior), size=m, replace=False)]) for _ in range(n_subsets)]
    return float(np.mean(scores))


def compute_pi_measures(
    author_papers: pd.DataFrame,
    embeddings: np.ndarray,
    vendi_m: int = 5,
    vendi_subsets: int = 20,
    recency_half_life: float = 2.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Computes the specialization measures for every paper, from the point of view of its PI.

    Output: one row per PI paper with
        paper_idx, pi_id, year, n_prior, career_years (years since the PI's first paper here),
        pivot, pivot_nn, pivot_recent, vendi.
    Measures are NaN when the PI has no prior papers (and vendi when fewer than vendi_m).
    """
    rng = np.random.default_rng(seed)
    decay = np.log(2.0) / recency_half_life
    records = []

    for author_id, group in author_papers.groupby("author_id"):
        if not group["is_pi"].any():
            continue

        paper_indices = group["paper_idx"].to_numpy()
        years = group["year"].to_numpy()
        first_year = years.min()
        pi_papers = group[group["is_pi"]]

        for paper_idx, year in zip(pi_papers["paper_idx"], pi_papers["year"]):
            is_prior = years < year
            prior = embeddings[paper_indices[is_prior]]
            paper = embeddings[paper_idx]

            record = {
                "paper_idx": paper_idx,
                "pi_id": author_id,
                "year": year,
                "n_prior": len(prior),
                "career_years": year - first_year,
                "pivot": np.nan,
                "pivot_nn": np.nan,
                "pivot_recent": np.nan,
                "vendi": np.nan,
            }

            if len(prior) > 0:
                prior_age = year - years[is_prior]
                record["pivot"] = cosine_distance_to_mean(paper, prior)
                record["pivot_nn"] = 1.0 - np.max(prior @ paper)
                record["pivot_recent"] = cosine_distance_to_mean(paper, prior, weights=np.exp(-decay * prior_age))
                record["vendi"] = rarefied_vendi(prior, vendi_m, vendi_subsets, rng)

            records.append(record)

    return pd.DataFrame(records)
