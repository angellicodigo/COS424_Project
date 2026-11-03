import pandas as pd


def parse_authors(raw_authors: str) -> list[str]:
    """Parse comma-separated author string into a list of cleaned names."""
    if not isinstance(raw_authors, str):
        return []
    return [name.strip() for name in raw_authors.split(",") if name.strip()]


def extract_last_author(raw_authors: str) -> str | None:
    """Return the last author name from an author string, or None if unavailable."""
    authors = parse_authors(raw_authors)
    return authors[-1] if authors else None


def extract_collaborators(raw_authors: str, pi: str) -> list[str]:
    """Return the co-authors on a paper excluding the specified last author (PI)."""
    authors = parse_authors(raw_authors)
    return [author for author in authors if author != pi]


def to_string_list(items: object) -> list[str]:
    """Convert an iterable or array into a list of strings."""
    if items is None:
        return []
    if hasattr(items, "tolist"):
        return [str(x) for x in items.tolist()]
    if isinstance(items, (list, tuple)):
        return [str(x) for x in items]
    return [str(items)]


def to_score_list(raw_scores: object) -> list[int]:
    """Convert raw paper scores from a numpy array or collection into a list of integers."""
    if raw_scores is None:
        return []
    if hasattr(raw_scores, "tolist"):
        return [int(x) for x in raw_scores.tolist()]
    if isinstance(raw_scores, (list, tuple)):
        return [int(x) for x in raw_scores]
    return []


def group_by_last_author(
    papers_df: pd.DataFrame,
    author_col: str = "authors",
    id_col: str = "id",
    title_col: str = "title",
    abstract_col: str = "abstract",
    decision_col: str = "decision",
    scores_col: str = "scores",
    keywords_col: str = "keywords",
    labels_col: str = "labels",
) -> pd.DataFrame:
    """Create a DataFrame mapping each last author to their papers' attributes."""
    last_authors = papers_df[author_col].apply(extract_last_author)
    # Recent papers may not list their authors because they are going through a double-blind review process
    valid_mask = last_authors.notna()

    paper_subset = pd.DataFrame(
        {
            "last_author": last_authors[valid_mask],
            "id": papers_df.loc[valid_mask, id_col],
            "title": papers_df.loc[valid_mask, title_col],
            "abstract": papers_df.loc[valid_mask, abstract_col],
            "collaborators": [
                extract_collaborators(author_str, pi_name)
                for author_str, pi_name in zip(
                    papers_df.loc[valid_mask, author_col],
                    last_authors[valid_mask],
                )
            ],
            "decision": papers_df.loc[valid_mask, decision_col],
            "score": papers_df.loc[valid_mask, scores_col].apply(to_score_list),
            "keywords": papers_df.loc[valid_mask, keywords_col].apply(to_string_list),
            "labels": papers_df.loc[valid_mask, labels_col],
        }
    )

    ids_by_last_authors = paper_subset.groupby("last_author").agg(
        ids=("id", list),
        titles=("title", list),
        abstracts=("abstract", list),
        collaborators=("collaborators", list),
        decisions=("decision", list),
        scores=("score", list),
        keywords=("keywords", list),
        labels=("labels", list),
        num_papers=("id", "count"),
    ).reset_index()

    return ids_by_last_authors
