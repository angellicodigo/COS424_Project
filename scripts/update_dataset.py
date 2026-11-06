from collections import Counter
from collections.abc import Sequence
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd

try:
    import openreview
    import openreview.api
except ImportError:
    openreview = None


# ---------------------------------------------------------------------------
# Default Keyword to Topic Mapping (from 21-rgm-create-labels-2026v1.ipynb)
# ---------------------------------------------------------------------------
LABEL_BY_KEYWORD: dict[str, str] = {
    # ADVERSARIAL
    "adversarial": "adversarial",
    "adversarial attack": "adversarial",
    "adversarial attacks": "adversarial",
    "adversarial defense": "adversarial",
    "adversarial examples": "adversarial",
    "adversarial example": "adversarial",
    "adversarial learning": "adversarial",
    "adversarial machine learning": "adversarial",
    "adversarial robustness": "adversarial",
    "adversarial training": "adversarial",
    # TRANSFORMERS
    "attention": "transformers",
    "attention mechanism": "transformers",
    "transformer": "transformers",
    "transformers": "transformers",
    "self-attention": "transformers",
    # AUTOENCODERS
    "autoencoder": "autoencoders",
    "autoencoders": "autoencoders",
    "vae": "autoencoders",
    "vaes": "autoencoders",
    "variational autoencoder": "autoencoders",
    "variational autoencoders": "autoencoders",
    # ANOMALY DETECTION
    "anomaly detection": "anomaly detection",
    # CAUSALITY
    "causal discovery": "causality",
    "causal inference": "causality",
    "causality": "causality",
    # CLUSTERING
    "clustering": "clustering",
    # COMPRESSION
    "compression": "compression",
    "model compression": "compression",
    # OBJECT DETECTION
    "object detection": "object detection",
    "semantic segmentation": "object detection",
    # CNNs
    "convolutional neural network": "CNNs",
    "convolutional neural networks": "CNNs",
    "cnn": "CNNs",
    "cnns": "CNNs",
    # DIFFUSION MODELS
    "diffusion": "diffusion models",
    "diffusion model": "diffusion models",
    "diffusion models": "diffusion models",
    # EXPLAINABILITY
    "explainability": "explainability",
    "explainable ai": "explainability",
    # INTERPRETABILITY
    "interpretability": "interpretability",
    # FAIRNESS
    "fairness": "fairness",
    # FEDERATED LEARNING
    "federated learning": "federated learning",
    # GANs
    "generative adversarial network": "GANs",
    "generative adversarial networks": "GANs",
    "gan": "GANs",
    "gans": "GANs",
    # GRAPHS
    "graph": "graphs",
    "graphs": "graphs",
    "graph neural network": "graphs",
    "graph neural networks": "graphs",
    "graph representation learning": "graphs",
    "gnn": "graphs",
    "gnns": "graphs",
    "node classification": "graphs",
    # LLMs
    "llm": "LLMs",
    "large language model": "LLMs",
    "large language models": "LLMs",
    "prompting": "LLMs",
    "bert": "LLMs",
    "llms": "LLMs",
    "text generation": "LLMs",
    # KNOWLEDGE DISTILLATION
    "knowledge distillation": "knowledge distillation",
    # LANGUAGE MODELS
    "natural language processing": "language models",
    "nlp": "language models",
    "language model": "language models",
    "language models": "language models",
    "language modeling": "language models",
    "machine translation": "language models",
    "question answering": "language models",
    "reasoning": "language models",
    # META LEARNING
    "meta learning": "meta learning",
    "meta-learning": "meta learning",
    # PRUNING
    "network pruning": "pruning",
    "pruning": "pruning",
    # NEURAL ARCHITECTURE SEARCH
    "neural architecture search": "neural architecture search",
    # OPTIMAL TRANSPORT
    "optimal transport": "optimal transport",
    # OPTIMIZATION
    "stochastic gradient descent": "optimization",
    "stochastic optimization": "optimization",
    "sgd": "optimization",
    "optimization": "optimization",
    "non-convex optimization": "optimization",
    "convex optimization": "optimization",
    "gradient descent": "optimization",
    "combinatorial optimization": "optimization",
    "bayesian optimization": "optimization",
    # OUT-OF-DISTRIBUTION
    "out-of-distribution": "out-of-distribution",
    "out-of-distribution detection": "out-of-distribution",
    "out-of-distribution generalization": "out-of-distribution",
    "distribution shift": "out-of-distribution",
    # PRIVACY
    "differential privacy": "privacy",
    "privacy": "privacy",
    # RNNs
    "rnn": "RNNs",
    "rnns": "RNNs",
    "recurrent neural network": "RNNs",
    "recurrent neural networks": "RNNs",
    "lstm": "RNNs",
    # REINFORCEMENT LEARNING
    "reinforcement learning": "RL",
    "deep reinforcement learning": "RL",
    "active learning": "active learning",
    "model-based reinforcement learning": "model-based RL",
    "multi-agent reinforcement learning": "multi-agent RL",
    "multi-agent": "multi-agent RL",
    "multi-task learning": "multi-task learning",
    "imitation learning": "imitation learning",
    # OFFLINE RL
    "offline reinforcement learning": "offline RL",
    "offline rl": "offline RL",
    # CONTINUAL LEARNING
    "continual learning": "continual learning",
    "lifelong learning": "continual learning",
    # IN-CONTEXT & FEW-SHOT
    "in-context learning": "in-context learning",
    "few-shot learning": "few-shot learning",
    # ROBUSTNESS & SSL
    "robustness": "robustness",
    "self-supervised learning": "self-supervised learning",
    "contrastive learning": "self-supervised learning",
    "semi-supervised learning": "semi-supervised learning",
    # TIME SERIES & TRANSFER LEARNING
    "time series": "time series",
    "time series forecasting": "time series",
    "transfer learning": "transfer learning",
    "domain adaptation": "transfer learning",
    "domain generalization": "transfer learning",
    # ViTs & VISION-LANGUAGE MODELS
    "vision transformer": "ViTs",
    "vision transformers": "ViTs",
    "vision-language models": "vision-language models",
    "vision-language model": "vision-language models",
    "clip": "vision-language models",
    # SAFETY & ALIGNMENT
    "ai safety": "safety",
    "safety": "safety",
    "alignment": "alignment",
    "rlhf": "alignment",
    "autonomous driving": "autonomous driving",
    "code generation": "code generation",
    "program synthesis": "code generation",
    "knowledge graph": "knowledge graphs",
    "knowledge graphs": "knowledge graphs",
    "neuroscience": "neuroscience",
    # 2026 CATEGORIES
    "3d reconstruction": "3D scenes",
    "novel view synthesis": "3D scenes",
    "nerf": "3D scenes",
    "gaussian splatting": "3D scenes",
    "speech synthesis": "speech",
    "text-to-speech": "speech",
    "speech recognition": "speech",
    "drug discovery": "molecules",
    "molecule generation": "molecules",
    "partial differential equations": "PDEs",
    "dynamical systems": "PDEs",
    "pdes": "PDEs",
    "pde": "PDEs",
}


# ---------------------------------------------------------------------------
# OpenReview Client & Content Extraction Helpers
# ---------------------------------------------------------------------------
def get_openreview_client(
    client: Any | None = None,
    baseurl: str = "https://api2.openreview.net",
    username: str | None = None,
    password: str | None = None,
    token: str | None = None,
) -> Any:
    """
    Initializes or returns an openreview.api.OpenReviewClient instance.

    Input:
        client: Pre-instantiated OpenReviewClient instance.
        baseurl: OpenReview API host URL (default: https://api2.openreview.net).
        username: Optional OpenReview account username/email.
        password: Optional OpenReview account password.
        token: Optional OpenReview bearer token.

    Output:
        client: Configured openreview.api.OpenReviewClient instance.
    """
    if client is not None:
        return client

    if openreview is None:
        raise ImportError(
            "The 'openreview' package is required for remote API fetching. "
            "Please install it via 'pip install openreview-py' or update your conda environment."
        )

    kwargs: dict[str, Any] = {"baseurl": baseurl}
    if token:
        kwargs["token"] = token
    elif username and password:
        kwargs["username"] = username
        kwargs["password"] = password

    return openreview.api.OpenReviewClient(**kwargs)


def extract_content_value(content_dict: dict[str, Any], key: str) -> Any:
    """
    Extracts a value from note content, accommodating both API v2 {'value': ...} dicts and raw primitives.

    Input:
        content_dict: Dictionary containing note content fields.
        key: Target field name string.

    Output:
        value: Extracted field value or None if key is absent.
    """
    if not isinstance(content_dict, dict) or key not in content_dict:
        return None
    val = content_dict[key]
    if isinstance(val, dict) and "value" in val:
        return val["value"]
    return val


def get_note_content(note: Any) -> dict[str, Any]:
    """
    Extracts content dictionary from either an openreview.api.Note object or a standard dictionary.

    Input:
        note: Note object or dictionary representing an OpenReview note.

    Output:
        content: Dictionary containing note content fields.
    """
    if hasattr(note, "content"):
        return note.content or {}
    if isinstance(note, dict):
        return note.get("content", {}) or {}
    return {}


def get_note_replies(note: Any) -> list[Any]:
    """
    Extracts direct reply notes (reviews, decisions) from an OpenReview Note object or dict.

    Input:
        note: OpenReview Note object or dictionary.

    Output:
        replies: List of direct reply Note objects or dictionaries.
    """
    if hasattr(note, "details") and isinstance(note.details, dict):
        return note.details.get("directReplies", [])
    if isinstance(note, dict):
        return note.get("details", {}).get("directReplies", [])
    return []


def fetch_openreview_forum_notes(
    forum_id: str,
    client: Any | None = None,
) -> list[Any]:
    """
    Fetches all notes associated with a forum ID using the openreview package.

    Input:
        forum_id: Paper forum identifier string.
        client: Optional OpenReviewClient instance.

    Output:
        notes: List of Note objects returned by the API.
    """
    openreview_client = get_openreview_client(client)
    try:
        return openreview_client.get_notes(forum=forum_id)
    except Exception:
        return []


def fetch_openreview_venue_notes(
    year: int = 2026,
    client: Any | None = None,
    include_replies: bool = True,
) -> dict[str, Any]:
    """
    Scrapes submissions, authors, decisions, and review ratings in bulk (~20 queries total).
    By specifying details='directReplies', OpenReview bundles all child reviews and decisions
    directly with each submission, eliminating the 500-request hourly rate limit.

    Input:
        year: Conference year to scrape.
        client: Optional OpenReviewClient instance.
        include_replies: Whether to fetch bundled direct replies (reviews and decisions).

    Output:
        noteByForumId: Dictionary mapping forum IDs to full Note objects.
    """
    openreview_client = get_openreview_client(client)
    noteByForumId: dict[str, Any] = {}
    queries = [
        "submission",
        "Submission",
        "Blind_Submission",
        "Withdrawn_Submission",
        "Rejected_Submission",
        "Desk_Rejected_Submission",
        "",
    ]
    details_param = "directReplies" if include_replies else None

    for query in queries:
        subquery = f"/{query}" if query else ""
        venue_id = f"ICLR.cc/{year}/Conference{subquery}"
        kwargs: dict[str, Any] = {"content": {"venueid": venue_id}}
        if details_param:
            kwargs["details"] = details_param
        try:
            notes = openreview_client.get_all_notes(**kwargs)
            for note in notes:
                forum_id = getattr(note, "forum", None) or (
                    note.get("forum") if isinstance(note, dict) else None
                )
                if forum_id and forum_id not in noteByForumId:
                    noteByForumId[forum_id] = note
        except Exception:
            continue

    return noteByForumId


def extract_authors_from_notes(notes: Sequence[Any]) -> tuple[str, str]:
    """
    Extracts author names and IDs from a collection of OpenReview Note objects or dicts.

    Input:
        notes: Sequence of Note objects for a single paper forum.

    Output:
        authors_and_ids: Tuple of (comma-separated author names, comma-separated author IDs).
    """
    for note in notes:
        content = get_note_content(note)
        raw_authors = extract_content_value(content, "authors")
        if raw_authors:
            raw_ids = extract_content_value(content, "authorids") or []
            author_str = (
                ", ".join(raw_authors)
                if isinstance(raw_authors, list)
                else str(raw_authors)
            )
            author_id_str = (
                ", ".join(raw_ids)
                if isinstance(raw_ids, list)
                else str(raw_ids or "")
            )
            return author_str, author_id_str
    return "", ""


def extract_decision_from_notes(notes: Sequence[Any]) -> str:
    """
    Extracts final review decision from forum notes replicating the notebook heuristic.

    Input:
        notes: Sequence of Note objects for a single paper forum.

    Output:
        decision: Extracted decision string (e.g. 'Accept', 'Reject', 'Withdrawn', 'Desk rejected').
    """
    for note in notes:
        content = get_note_content(note)
        dec_val = extract_content_value(content, "decision")
        if dec_val:
            return str(dec_val)
        if "withdrawal_confirmation" in content:
            return "Withdrawn"
        if "desk_reject_comments" in content:
            return "Desk rejected"
    return ""


def extract_scores_from_notes(notes: Sequence[Any]) -> list[int]:
    """
    Extracts reviewer rating scores from forum notes replicating the original scraper.

    Input:
        notes: Sequence of Note objects for a single paper forum.

    Output:
        scores: List of integer reviewer scores.
    """
    scores: list[int] = []
    for note in notes:
        content = get_note_content(note)
        rating_val = extract_content_value(content, "rating")
        if rating_val is not None:
            try:
                scores.append(int(rating_val))
            except (ValueError, TypeError):
                continue
    return scores


def compute_keyword_frequencies(
    keywords_series: Sequence[Sequence[str]],
) -> dict[str, int]:
    """
    Computes global frequency counts of lowercased keywords across paper records.

    Input:
        keywords_series: Sequence of keyword collections from paper entries.

    Output:
        countByKeyword: Dictionary mapping each lowercase keyword to its occurrence count.
    """
    countByKeyword: dict[str, int] = Counter()
    for paper_keywords in keywords_series:
        if isinstance(paper_keywords, (list, tuple, np.ndarray)):
            for kw in paper_keywords:
                cleaned = str(kw).strip().lower()
                if cleaned:
                    countByKeyword[cleaned] += 1
    return dict(countByKeyword)


def assign_single_label(
    keywords: Sequence[str],
    countByKeyword: dict[str, int],
    labelByKeyword: dict[str, str],
) -> str:
    """
    Selects the most specific topic label by finding the minimum frequency keyword.

    Input:
        keywords: Sequence of keyword strings for a single paper.
        countByKeyword: Dictionary mapping keywords to their dataset-wide frequencies.
        labelByKeyword: Dictionary mapping keywords to topic classification labels.

    Output:
        label: Selected topic category string or 'unlabeled' if no matching keyword.
    """
    if not isinstance(keywords, (list, tuple, np.ndarray)) or len(keywords) == 0:
        return "unlabeled"

    matched_keywords = [
        str(kw).strip().lower()
        for kw in keywords
        if str(kw).strip().lower() in labelByKeyword
    ]
    if not matched_keywords:
        return "unlabeled"

    # Select the keyword with the lowest frequency in corpus (most specific topic)
    chosen_keyword = min(
        matched_keywords, key=lambda k: countByKeyword.get(k, float("inf"))
    )
    return labelByKeyword.get(chosen_keyword, "unlabeled")


def format_update_message(entity_name: str, updated_count: int) -> str:
    """
    Formats a human-readable update message with correct singular or plural phrasing.

    Input:
        entity_name: Subject being updated (e.g., 'authors', 'decisions', 'review scores', 'topic labels').
        updated_count: Integer number of records updated.

    Output:
        message: Formatted summary string.
    """
    noun = "paper" if updated_count == 1 else "papers"
    return f"Updated {entity_name} for {updated_count} {noun}."


# ---------------------------------------------------------------------------
# Core Update Routines
# ---------------------------------------------------------------------------
def update_authors(
    papers_df: pd.DataFrame,
    year: int = 2026,
    authors_by_id: dict[str, tuple[str, str]] | None = None,
    fetch_remote: bool = False,
    client: Any | None = None,
    venue_notes: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Updates papers that currently have empty ("") or "Anonymous" author values.

    Input:
        papers_df: DataFrame containing paper records with 'id' and 'authors' columns.
        year: Conference year targeted for updates.
        authors_by_id: Optional lookup dictionary mapping paper ID to (authors, author_ids).
        fetch_remote: Whether to query OpenReview API for missing author records.
        client: Optional OpenReviewClient instance.
        venue_notes: Optional pre-fetched bulk venue Note mapping from fetch_openreview_venue_notes.

    Output:
        updated_df: Updated DataFrame copy with populated author fields.
    """
    updated_df = papers_df.copy()
    author_str_series = updated_df["authors"].fillna("").astype(str).str.strip()
    is_missing_author = (author_str_series == "") | (
        author_str_series.str.lower() == "anonymous"
    )

    updated_count = 0

    if not is_missing_author.any():
        print(format_update_message("authors", updated_count))
        return updated_df

    # Populate from explicit lookup dictionary if provided
    if authors_by_id is not None:
        for idx in updated_df[is_missing_author].index:
            paper_id = updated_df.at[idx, "id"]
            if paper_id in authors_by_id:
                new_authors, new_ids = authors_by_id[paper_id]
                updated_df.at[idx, "authors"] = new_authors
                if "author_ids" in updated_df.columns:
                    updated_df.at[idx, "author_ids"] = new_ids
                updated_count += 1
        print(format_update_message("authors", updated_count))
        return updated_df

    # Fetch remotely using OpenReviewClient bulk venue query
    if fetch_remote:
        notes_map = (
            venue_notes
            if venue_notes is not None
            else fetch_openreview_venue_notes(
                year=year, client=client, include_replies=False
            )
        )
        for idx in updated_df[is_missing_author].index:
            paper_id = updated_df.at[idx, "id"]
            if paper_id in notes_map:
                note = notes_map[paper_id]
                content = get_note_content(note)
                raw_authors = extract_content_value(content, "authors")
                if raw_authors:
                    raw_ids = extract_content_value(content, "authorids") or []
                    updated_df.at[idx, "authors"] = (
                        ", ".join(raw_authors)
                        if isinstance(raw_authors, list)
                        else str(raw_authors)
                    )
                    if "author_ids" in updated_df.columns:
                        updated_df.at[idx, "author_ids"] = (
                            ", ".join(raw_ids)
                            if isinstance(raw_ids, list)
                            else str(raw_ids or "")
                        )
                    updated_count += 1

    print(format_update_message("authors", updated_count))
    return updated_df


def update_decisions(
    papers_df: pd.DataFrame,
    decision_by_id: dict[str, str] | None = None,
    fetch_remote: bool = False,
    client: Any | None = None,
    venue_notes: dict[str, Any] | None = None,
    forum_cache: dict[str, list[Any]] | None = None,
) -> pd.DataFrame:
    """
    Updates papers that are missing acceptance or rejection decisions.

    Input:
        papers_df: DataFrame containing paper records with 'id' and 'decision' columns.
        decision_by_id: Optional lookup dictionary mapping paper ID to decision string.
        fetch_remote: Whether to query OpenReview API for missing decisions.
        client: Optional OpenReviewClient instance.
        venue_notes: Optional pre-fetched bulk venue Note mapping from fetch_openreview_venue_notes.
        forum_cache: Optional shared cache dictionary mapping paper ID to forum note list.

    Output:
        updated_df: Updated DataFrame copy with populated decision strings.
    """
    updated_df = papers_df.copy()
    decision_str_series = updated_df["decision"].fillna("").astype(str).str.strip()
    is_missing_decision = decision_str_series == ""

    updated_count = 0

    if not is_missing_decision.any():
        print(format_update_message("decisions", updated_count))
        return updated_df

    # Populate from explicit lookup dictionary if provided
    if decision_by_id is not None:
        for idx in updated_df[is_missing_decision].index:
            paper_id = updated_df.at[idx, "id"]
            if paper_id in decision_by_id:
                updated_df.at[idx, "decision"] = decision_by_id[paper_id]
                updated_count += 1
        print(format_update_message("decisions", updated_count))
        return updated_df

    # Populate from bulk venue notes (bundled with directReplies) or individual forum notes
    if fetch_remote:
        cache = forum_cache if forum_cache is not None else {}
        for idx in updated_df[is_missing_decision].index:
            paper_id = updated_df.at[idx, "id"]
            decision = ""
            if venue_notes is not None and paper_id in venue_notes:
                note = venue_notes[paper_id]
                replies = get_note_replies(note)
                decision = extract_decision_from_notes(replies or [note])

            if not decision:
                if paper_id in cache:
                    notes = cache[paper_id]
                else:
                    notes = fetch_openreview_forum_notes(paper_id, client=client)
                    cache[paper_id] = notes
                decision = extract_decision_from_notes(notes)

            if decision:
                updated_df.at[idx, "decision"] = decision
                updated_count += 1

    print(format_update_message("decisions", updated_count))
    return updated_df


def update_scores(
    papers_df: pd.DataFrame,
    scores_by_id: dict[str, list[int]] | None = None,
    fetch_remote: bool = False,
    client: Any | None = None,
    venue_notes: dict[str, Any] | None = None,
    forum_cache: dict[str, list[Any]] | None = None,
) -> pd.DataFrame:
    """
    Updates papers that have empty review score lists.

    Input:
        papers_df: DataFrame containing paper records with 'id' and 'scores' columns.
        scores_by_id: Optional lookup dictionary mapping paper ID to list of integer scores.
        fetch_remote: Whether to query OpenReview API for missing reviewer scores.
        client: Optional OpenReviewClient instance.
        venue_notes: Optional pre-fetched bulk venue Note mapping from fetch_openreview_venue_notes.
        forum_cache: Optional shared cache dictionary mapping paper ID to forum note list.

    Output:
        updated_df: Updated DataFrame copy with populated review score lists.
    """
    updated_df = papers_df.copy()
    is_missing_scores = updated_df["scores"].apply(
        lambda s: len(s) == 0 if hasattr(s, "__len__") else True
    )

    updated_count = 0

    if not is_missing_scores.any():
        print(format_update_message("review scores", updated_count))
        return updated_df

    # Populate from explicit lookup dictionary if provided
    if scores_by_id is not None:
        for idx in updated_df[is_missing_scores].index:
            paper_id = updated_df.at[idx, "id"]
            if paper_id in scores_by_id:
                updated_df.at[idx, "scores"] = scores_by_id[paper_id]
                updated_count += 1
        print(format_update_message("review scores", updated_count))
        return updated_df

    # Populate from bulk venue notes (bundled with directReplies) or individual forum notes
    if fetch_remote:
        cache = forum_cache if forum_cache is not None else {}
        for idx in updated_df[is_missing_scores].index:
            paper_id = updated_df.at[idx, "id"]
            scores = []
            if venue_notes is not None and paper_id in venue_notes:
                note = venue_notes[paper_id]
                replies = get_note_replies(note)
                scores = extract_scores_from_notes(replies)

            if not scores:
                if paper_id in cache:
                    notes = cache[paper_id]
                else:
                    notes = fetch_openreview_forum_notes(paper_id, client=client)
                    cache[paper_id] = notes
                scores = extract_scores_from_notes(notes)

            if scores:
                updated_df.at[idx, "scores"] = scores
                updated_count += 1

    print(format_update_message("review scores", updated_count))
    return updated_df


def update_labels(
    papers_df: pd.DataFrame,
    label_by_keyword: dict[str, str] | None = None,
    overwrite_existing: bool = False,
) -> pd.DataFrame:
    """
    Updates topic labels for papers based on keyword specificity matching the original notebook methodology.

    Input:
        papers_df: DataFrame containing paper records with 'keywords' and 'labels' columns.
        label_by_keyword: Optional keyword-to-label mapping (defaults to LABEL_BY_KEYWORD).
        overwrite_existing: If True, recomputes all labels; if False, only computes empty or 'unlabeled' entries.

    Output:
        updated_df: Updated DataFrame copy with assigned topic labels.
    """
    updated_df = papers_df.copy()
    mapping = label_by_keyword or LABEL_BY_KEYWORD

    # Compute global keyword frequency across entire corpus
    countByKeyword = compute_keyword_frequencies(updated_df["keywords"])

    if overwrite_existing:
        target_indices = updated_df.index
    else:
        label_series = updated_df["labels"].fillna("").astype(str).str.strip()
        target_indices = updated_df[
            (label_series == "") | (label_series == "unlabeled")
        ].index

    updated_count = 0

    # Assign labels based on the minimum frequency keyword
    for idx in target_indices:
        keywords = updated_df.at[idx, "keywords"]
        old_label = updated_df.at[idx, "labels"]
        new_label = assign_single_label(keywords, countByKeyword, mapping)
        if new_label != old_label:
            updated_df.at[idx, "labels"] = new_label
            updated_count += 1

    print(format_update_message("topic labels", updated_count))
    return updated_df


def update(
    papers_df: pd.DataFrame,
    year: int = 2026,
    fetch_remote: bool = False,
    authors_by_id: dict[str, tuple[str, str]] | None = None,
    decision_by_id: dict[str, str] | None = None,
    scores_by_id: dict[str, list[int]] | None = None,
    label_by_keyword: dict[str, str] | None = None,
    overwrite_labels: bool = False,
    output_path: Path | str | None = None,
    output_name: str | None = None,
    client: Any | None = None,
    username: str | None = None,
    password: str | None = None,
    token: str | None = None,
) -> pd.DataFrame:
    """
    High-level orchestrator that sequentially updates authors, decisions, review scores, and topic labels.

    Input:
        papers_df: Input DataFrame containing ICLR paper records.
        year: Conference year targeted for updates.
        fetch_remote: Flag indicating whether to fetch unblinded metadata from OpenReview.
        authors_by_id: Optional dictionary of pre-fetched (authors, author_ids) by paper ID.
        decision_by_id: Optional dictionary of pre-fetched decisions by paper ID.
        scores_by_id: Optional dictionary of pre-fetched scores by paper ID.
        label_by_keyword: Optional keyword-to-label mapping dictionary.
        overwrite_labels: Flag indicating whether to overwrite existing topic labels.
        output_path: Optional destination directory or file path to save updated DataFrame as parquet.
        output_name: Optional file name (defaults to 'iclr_updated.parquet' or '<output_name>.parquet').
        client: Optional OpenReviewClient instance.
        username: Optional OpenReview account username/email.
        password: Optional OpenReview account password.
        token: Optional OpenReview bearer token.

    Output:
        updated_df: DataFrame containing updated paper records.
    """
    updated_df = papers_df.copy()
    forum_cache: dict[str, list[Any]] = {}

    openreview_client = None
    venue_notes = None

    if fetch_remote:
        openreview_client = get_openreview_client(
            client=client, username=username, password=password, token=token
        )
        print(f"Fetching ICLR {year} submissions, reviews, and decisions in bulk...")
        venue_notes = fetch_openreview_venue_notes(
            year=year, client=openreview_client, include_replies=True
        )

    updated_df = update_authors(
        updated_df,
        year=year,
        authors_by_id=authors_by_id,
        fetch_remote=fetch_remote,
        client=openreview_client,
        venue_notes=venue_notes,
    )
    updated_df = update_decisions(
        updated_df,
        decision_by_id=decision_by_id,
        fetch_remote=fetch_remote,
        client=openreview_client,
        venue_notes=venue_notes,
        forum_cache=forum_cache,
    )
    updated_df = update_scores(
        updated_df,
        scores_by_id=scores_by_id,
        fetch_remote=fetch_remote,
        client=openreview_client,
        venue_notes=venue_notes,
        forum_cache=forum_cache,
    )
    updated_df = update_labels(
        updated_df,
        label_by_keyword=label_by_keyword,
        overwrite_existing=overwrite_labels,
    )

    # Save to parquet if output_path is provided
    if output_path is not None:
        target_dir = Path(output_path)
        if output_name is not None:
            filename = (
                output_name
                if output_name.endswith(".parquet")
                else f"{output_name}.parquet"
            )
        else:
            filename = "iclr_updated.parquet"

        if target_dir.suffix == ".parquet" and output_name is None:
            target_file = target_dir
        else:
            target_file = target_dir / filename

        target_file.parent.mkdir(parents=True, exist_ok=True)
        updated_df.to_parquet(target_file, index=False)
        print(f"Saved updated dataset to: {target_file.resolve()}")

    return updated_df


if __name__ == "__main__":
    print("Testing OpenReview API connection using openreview-py...")
    if openreview is None:
        print("The 'openreview' package is not installed yet in this environment.")
    else:
        try:
            import importlib.metadata

            pkg_version = importlib.metadata.version("openreview-py")
            print(f"Using openreview-py version: {pkg_version}")
        except Exception:
            pass

        client = openreview.api.OpenReviewClient(baseurl="https://api2.openreview.net")
        print("Connecting to OpenReview API v2 (https://api2.openreview.net)...")
        try:
            notes = client.get_notes(
                content={"venueid": "ICLR.cc/2026/Conference/Submission"}, limit=1
            )
            print(f"Success! Notes retrieved: {len(notes)}")
            if notes:
                print(f"Sample Note ID: {notes[0].id}")
        except Exception as o_err:
            err_dict = getattr(o_err, "args", [None])[0] if getattr(o_err, "args", None) else {}
            status_code = err_dict.get("status", 403) if isinstance(err_dict, dict) else 403
            print(f"OpenReview API Error (status {status_code}): {o_err}")
            print(
                "\nNotice: Anonymous requests to /notes trigger OpenReview's Cloudflare challenge."
            )
            print(
                "To authenticate and query, instantiate with your OpenReview account:"
            )
            print(
                "  client = openreview.api.OpenReviewClient(baseurl='https://api2.openreview.net', username='...', password='...')"
            )
            print("  update(df, fetch_remote=True, client=client)")
