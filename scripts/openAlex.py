import requests
from typing import Any, Dict, List, Optional

OPENALEX_API_BASE = "https://api.openalex.org"


def create_headers(email: Optional[str] = None) -> Dict[str, str]:
    """Builds request headers, opting into the OpenAlex Polite Pool if an email is provided."""
    if email:
        return {"User-Agent": f"mailto:{email}"}
    return {"User-Agent": "COS424-Project/1.0"}


def search_work_by_title(
    title: str,
    email: Optional[str] = None
) -> Dict[str, Any]:
    """Searches OpenAlex for a single work matching the given title."""
    url = f"{OPENALEX_API_BASE}/works"
    params = {"search": title, "per_page": 1}
    headers = create_headers(email)

    response = requests.get(url, params=params, headers=headers)
    response.raise_for_status()

    results = response.json().get("results", [])
    if not results:
        raise LookupError(f"No paper found on OpenAlex matching title: {title!r}")

    return results[0]


def extract_author_id(
    work: Dict[str, Any],
    target_author_name: str
) -> Optional[str]:
    """Finds the OpenAlex author ID corresponding to target_author_name in a work."""
    target_clean = target_author_name.lower().strip()
    authorships = work.get("authorships", [])

    for authorship in authorships:
        author = authorship.get("author", {})
        display_name = author.get("display_name", "").lower().strip()
        if target_clean in display_name or display_name in target_clean:
            return author.get("id")

    return None


def extract_venue(work: Dict[str, Any]) -> str:
    """Extracts the venue or host journal/conference name from an OpenAlex work."""
    primary_location = work.get("primary_location") or {}
    source = primary_location.get("source") or {}
    return source.get("display_name") or ""


def is_iclr_paper(work: Dict[str, Any]) -> bool:
    """Checks whether a work was published at ICLR."""
    venue_name = extract_venue(work).lower()
    return "iclr" in venue_name or "international conference on learning representations" in venue_name


def fetch_works_by_author_id(
    author_id: str,
    email: Optional[str] = None,
    limit: int = 200
) -> List[Dict[str, Any]]:
    """Retrieves up to `limit` works authored by a given OpenAlex author ID."""
    url = f"{OPENALEX_API_BASE}/works"
    params = {"filter": f"author.id:{author_id}", "per_page": min(limit, 200)}
    headers = create_headers(email)

    response = requests.get(url, params=params, headers=headers)
    response.raise_for_status()

    return response.json().get("results", [])


def format_work_summary(work: Dict[str, Any]) -> Dict[str, Any]:
    """Formats an OpenAlex work record into a standardized paper summary dictionary."""
    return {
        "id": work.get("id"),
        "title": work.get("title"),
        "year": work.get("publication_year"),
        "venue": extract_venue(work),
        "doi": work.get("doi"),
        "cited_by_count": work.get("cited_by_count", 0),
        "type": work.get("type"),
    }


def fetch_papers_by_reference(
    paper_title: str,
    target_author: str,
    email: Optional[str] = "student@princeton.edu",
    limit: int = 200
) -> List[Dict[str, Any]]:
    """
    Finds an author's unique OpenAlex ID using a reference seed paper,
    then retrieves all their publications outside of ICLR.
    """
    seed_work = search_work_by_title(paper_title, email=email)
    author_id = extract_author_id(seed_work, target_author)

    if not author_id:
        return []

    author_works = fetch_works_by_author_id(author_id, email=email, limit=limit)
    external_works = [w for w in author_works if not is_iclr_paper(w)]

    return [format_work_summary(w) for w in external_works]


if __name__ == "__main__":
    title = "Randomly pivoted Cholesky: Practical approximation of a kernel matrix with few entry evaluations"
    target_author = "Robert J. Webber"
    papers = fetch_papers_by_reference(title, target_author)
    print(papers)
    

    print(f"Found {len(papers)} papers outside ICLR for {target_author}:")
    for paper in papers[:5]:
        print(f"[{paper['year']}] {paper['title']} - {paper['venue'] or 'Preprint/Unknown'}")
