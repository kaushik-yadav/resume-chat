from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


def test_health_reports_configuration():
    app = create_app(Settings(llm_provider="fake", storage_backend="memory"))
    response = TestClient(app).get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["llm_provider"] == "fake"
    assert body["storage_backend"] == "memory"
