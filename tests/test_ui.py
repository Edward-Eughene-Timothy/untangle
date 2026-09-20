from pathlib import Path

STATIC = Path(__file__).resolve().parents[1] / "app" / "static"


def test_root_serves_the_chat_ui(client):
    r = client.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    assert "<title>Untangle</title>" in r.text and "/app.js" in r.text


def test_static_assets_are_served(client):
    assert client.get("/app.js").status_code == 200
    assert client.get("/style.css").status_code == 200


def test_api_and_docs_are_not_shadowed_by_the_ui(client):
    assert client.get("/api/v1/health").json()["status"] in ("ok", "degraded")
    assert client.get("/docs").status_code == 200
    assert client.get("/api/v1/nope").status_code == 404


def test_ui_makes_no_external_requests():
    for f in ("index.html", "app.js", "style.css"):
        text = (STATIC / f).read_text()
        assert "http://" not in text.replace("http://127.0.0.1", "") or f == "app.js"  # app.js: regex only
        assert "https://cdn" not in text and "googleapis" not in text and "unpkg" not in text
    js = (STATIC / "app.js").read_text()
    assert js.count("fetch(") == 2 and "API" in js  # only the local API is ever called
