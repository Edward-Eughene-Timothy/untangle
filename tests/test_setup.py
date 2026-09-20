from app.services.warmup_service import chunk_text, html_to_text
from tests.conftest import fake_fetcher

API = "/api/v1"


def test_status_before_warmup(client):
    s = client.get(f"{API}/setup/status").json()
    assert s["state"] == "idle" and s["initialized"] is False


def test_topics_listed(client):
    ids = {t["id"] for t in client.get(f"{API}/setup/topics").json()}
    assert {"cbt", "ifs", "nvc", "decision_paralysis", "imposter_syndrome"} <= ids


def test_unknown_topic_rejected(client):
    r = client.post(f"{API}/setup/warmup", json={"topics": ["astrology"]})
    assert r.status_code == 400


def test_empty_topics_rejected(client):
    assert client.post(f"{API}/setup/warmup", json={"topics": []}).status_code == 422


def test_warmup_indexes_and_sets_flag(client):
    r = client.post(f"{API}/setup/warmup", json={"topics": ["cbt", "ifs"]})
    assert r.status_code == 202
    s = client.get(f"{API}/setup/status").json()
    assert s["state"] == "completed"
    assert s["initialized"] is True
    assert s["pages_done"] == s["pages_total"] == 5
    assert s["chunks_indexed"] > 5
    assert client.get(f"{API}/health").json()["qdrant"]["chunks"] == s["chunks_indexed"]


def test_warmup_is_idempotent(client):
    client.post(f"{API}/setup/warmup", json={"topics": ["ifs"]})
    first = client.get(f"{API}/health").json()["qdrant"]["chunks"]
    client.post(f"{API}/setup/warmup", json={"topics": ["ifs"]})
    assert client.get(f"{API}/health").json()["qdrant"]["chunks"] == first


def test_partial_failure_still_initializes(make_client):
    with make_client(fetcher=fake_fetcher) as c:
        r = c.post(
            f"{API}/setup/warmup",
            json={"topics": ["ifs"], "extra_urls": ["https://example.com/FAIL"]},
        )
        assert r.status_code == 202
        s = c.get(f"{API}/setup/status").json()
        assert s["state"] == "completed" and s["pages_failed"] == 1
        assert s["failed_urls"] == ["https://example.com/FAIL"]


def test_total_failure_does_not_initialize(make_client):
    async def always_fail(url):
        raise ConnectionError("offline")

    with make_client(fetcher=always_fail) as c:
        c.post(f"{API}/setup/warmup", json={"topics": ["ifs"]})
        s = c.get(f"{API}/setup/status").json()
        assert s["state"] == "failed" and s["initialized"] is False
        assert "offline" in s["error"]


def test_initialized_flag_survives_restart(make_client):
    with make_client() as c1:
        c1.post(f"{API}/setup/warmup", json={"topics": ["ifs"]})
    with make_client() as c2:
        s = c2.get(f"{API}/setup/status").json()
        assert s["initialized"] is True
        assert c2.get(f"{API}/health").json()["qdrant"]["chunks"] > 0


# ---------------------------------------------------------------- pure unit tests


def test_chunk_text_respects_size_and_overlap():
    text = " ".join(f"Sentence number {i} is here." for i in range(200))
    chunks = chunk_text(text, size=500, overlap=50)
    assert len(chunks) > 5
    assert all(len(c) <= 500 for c in chunks)
    # consecutive chunks share some text
    assert all(set(a.split()[-3:]) & set(b.split()[:12]) for a, b in zip(chunks, chunks[1:]))
    # nothing is lost: every sentence appears somewhere
    joined = " ".join(chunks)
    assert all(f"Sentence number {i} is" in joined for i in range(200))


def test_chunk_text_short_input():
    assert chunk_text("Just one short sentence that is long enough to keep.") != []
    assert chunk_text("   ") == []


def test_html_to_text_strips_chrome_and_references():
    body = "Real content paragraph that is long enough to be kept by the extractor. " * 3
    html = f"""
    <html><head><title>T</title><script>evil()</script></head><body>
    <nav>menu menu menu menu menu menu menu menu</nav>
    <h1>Main Title</h1>
    <div id="mw-content-text">
      <p>{body}<sup class="reference">[1]</sup></p>
      <div class="navbox"><p>navigation box text that should vanish entirely</p></div>
      <h2>Methods</h2><p>Another substantive paragraph about methods and techniques used.</p>
      <h2>References</h2><p>Smith, J. (2001). A citation that must not be indexed at all.</p>
    </div></body></html>"""
    title, text = html_to_text(html)
    assert title == "Main Title"
    assert "Real content paragraph" in text and "Methods." in text
    assert "menu" not in text and "evil" not in text
    assert "navigation box" not in text and "Smith" not in text and "[1]" not in text
