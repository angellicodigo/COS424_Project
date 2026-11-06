import pandas as pd


def parse_authors(raw_authors: str) -> list[str]:
    """
    Parses comma-separated author string into a cleaned list of author names.

    Input:
        raw_authors: Comma-separated string containing author names.

    Output:
        authors: List of trimmed, non-empty author name strings.
    """
    if not isinstance(raw_authors, str):
        return []
    # Split by comma, trim whitespace, and filter out empty strings
    return [name.strip() for name in raw_authors.split(",") if name.strip()]


def extract_last_author(raw_authors: str) -> str | None:
    """
    Extracts the last author name from an author string (assumed to be the Principal Investigator).

    Input:
        raw_authors: Comma-separated string containing author names.

    Output:
        last_author: Name of the last author, or None if no valid author is found.
    """
    authors = parse_authors(raw_authors)  # extract cleaned list of authors
    return authors[-1] if authors else None  # return the last listed author or None


def extract_collaborators(raw_authors: str, pi: str) -> list[str]:
    """
    Extracts the co-authors on a paper excluding the specified last author (PI).

    Input:
        raw_authors: Comma-separated string containing author names.
        pi: Name of the principal investigator / last author to exclude.

    Output:
        collaborators: List of co-author names excluding the PI.
    """
    authors = parse_authors(raw_authors)  # parse into list of author names
    return [author for author in authors if author != pi]  # filter out the PI


def to_string_list(items: object) -> list[str]:
    """
    Converts an iterable, numpy array, collection, or scalar into a standardized list of strings.

    Input:
        items: An iterable, numpy array, list, tuple, or scalar value.

    Output:
        string_list: Standardized list of string elements.
    """
    if items is None:
        return []
    if hasattr(items, "tolist"):
        return [str(x) for x in items.tolist()]  # convert numpy array to list of strings
    if isinstance(items, (list, tuple)):
        return [str(x) for x in items]  # convert collection elements to strings
    return [str(items)]  # wrap single scalar item in a list


def to_score_list(raw_scores: object) -> list[int]:
    """
    Converts raw paper review scores from a collection or numpy array into a list of integers.

    Input:
        raw_scores: An iterable, numpy array, list, or tuple containing review score values.

    Output:
        score_list: List of integer score values.
    """
    if raw_scores is None:
        return []
    if hasattr(raw_scores, "tolist"):
        return [int(x) for x in raw_scores.tolist()]  # convert numpy array to integer list
    if isinstance(raw_scores, (list, tuple)):
        return [int(x) for x in raw_scores]  # convert collection elements to integers
    return []


def extract_unique_last_authors(raw_authors_series: pd.Series) -> set[str]:
    """
    Extracts the unique set of non-empty last-listed authors across all papers.

    Input:
        raw_authors_series: Pandas Series containing comma-separated author strings.

    Output:
        unique_last_authors: Set of unique last-author name strings.
    """
    return {
        last_author
        for raw in raw_authors_series.dropna()
        if (last_author := extract_last_author(raw)) and last_author != "Anonymous"
    }


def build_pi_paper_records(
    papers_df: pd.DataFrame,
    target_authors: set[str],
    author_col: str,
    id_col: str,
    title_col: str,
    abstract_col: str,
    decision_col: str,
    scores_col: str,
    keywords_col: str,
    labels_col: str,
) -> pd.DataFrame:
    """
    Builds a normalized DataFrame mapping each targeted author to all of their co-authored papers.

    Input:
        papers_df: DataFrame containing paper records and metadata.
        target_authors: Set of author names to extract papers for.
        author_col: Column name for authors.
        id_col: Column name for paper id.
        title_col: Column name for title.
        abstract_col: Column name for abstract.
        decision_col: Column name for decision.
        scores_col: Column name for scores.
        keywords_col: Column name for keywords.
        labels_col: Column name for labels.

    Output:
        paper_records_df: DataFrame containing flattened records per targeted author and paper.
    """
    prepared_df = pd.DataFrame(
        {
            "authors": papers_df[author_col].apply(parse_authors),
            "id": papers_df[id_col],
            "title": papers_df[title_col],
            "abstract": papers_df[abstract_col],
            "decision": papers_df[decision_col],
            "scores": papers_df[scores_col].apply(to_score_list),
            "keywords": papers_df[keywords_col].apply(to_string_list),
            "labels": papers_df[labels_col],
        }
    )

    records = []
    for row in prepared_df.itertuples(index=False):
        matching_pis = [author for author in row.authors if author in target_authors]
        for pi in matching_pis:
            collaborators = [author for author in row.authors if author != pi]
            records.append(
                (
                    pi,
                    row.id,
                    row.title,
                    row.abstract,
                    collaborators,
                    row.decision,
                    row.scores,
                    row.keywords,
                    row.labels,
                )
            )

    column_names = [
        "last_author",
        "id",
        "title",
        "abstract",
        "collaborators",
        "decision",
        "score",
        "keywords",
        "labels",
    ]
    return pd.DataFrame(records, columns=column_names)


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
    """
    Groups papers by authors who have appeared as a last author (PI), aggregating all papers they co-authored.

    Input:
        papers_df: DataFrame containing paper records and metadata.
        author_col: Name of column containing comma-separated author strings.
        id_col: Name of column containing paper identifiers.
        title_col: Name of column containing paper titles.
        abstract_col: Name of column containing paper abstracts.
        decision_col: Name of column containing paper review decisions.
        scores_col: Name of column containing review scores.
        keywords_col: Name of column containing paper keywords.
        labels_col: Name of column containing topic or classification labels.

    Output:
        ids_by_last_authors: Aggregated DataFrame where each row corresponds to a PI with lists of paper attributes and total publication count across all co-authored papers.
    """
    unique_last_authors = extract_unique_last_authors(papers_df[author_col])
    paper_subset = build_pi_paper_records(
        papers_df=papers_df,
        target_authors=unique_last_authors,
        author_col=author_col,
        id_col=id_col,
        title_col=title_col,
        abstract_col=abstract_col,
        decision_col=decision_col,
        scores_col=scores_col,
        keywords_col=keywords_col,
        labels_col=labels_col,
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

