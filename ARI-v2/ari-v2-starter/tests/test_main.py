from app.main import Paper, ResearchRequest, health


def test_health():
    assert health()["status"] == "ok"


def test_request_defaults():
    assert ResearchRequest(question="RF modulation classification").limit == 10


def test_paper_defaults():
    paper = Paper(title="Test", paper_id="123")
    assert paper.citation_count == 0
    assert paper.relevance == 0.0
