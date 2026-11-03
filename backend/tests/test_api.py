"""API integration tests for Skinora FastAPI application."""
import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image

try:
    from Skinora.backend.app.main import app
except ImportError:
    from backend.app.main import app

client = TestClient(app)


def test_root_endpoint():
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["app"] == "Skinora API"
    assert data["status"] == "running"


def test_health_endpoint():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "acne_model" in data


def test_model_status_endpoint():
    resp = client.get("/api/model-status")
    assert resp.status_code == 200
    data = resp.json()
    assert "available" in data
    assert "status" in data
    assert "classes" in data
    assert len(data["classes"]) == 4


def test_model_evaluation_endpoint():
    resp = client.get("/api/model-evaluation")
    assert resp.status_code == 200
    data = resp.json()
    assert "available" in data
    if data["available"]:
        assert "report" in data
        assert "confusion_matrix" in data["report"]
        assert "accuracy" in data["report"]
        assert "macro_f1" in data["report"]


def test_analyze_questionnaire_only():
    data = {
        "skin_type": "Dry",
        "age_range": "26–35",
        "concerns": '["Dryness", "Uneven texture"]',
        "sensitivity": "Rarely",
        "severity": 3,
    }
    resp = client.post("/api/analyze", data=data)
    assert resp.status_code == 200
    body = resp.json()
    assert body["skin_type"] == "Dry"
    assert body["photo_quality"]["uploaded"] is False
    assert body["acne_prediction"]["status"] == "no_photo"
    assert "routine" in body
    assert "morning" in body["routine"]


def test_analyze_with_synthetic_photo():
    img = Image.new("RGB", (224, 224), color=(210, 180, 160))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    data = {
        "skin_type": "Combination",
        "age_range": "18–25",
        "concerns": '["Acne & blemishes"]',
        "sensitivity": "Sometimes",
        "severity": 2,
    }
    files = {"photo": ("test.jpg", buf.getvalue(), "image/jpeg")}
    resp = client.post("/api/analyze", data=data, files=files)
    assert resp.status_code == 200
    body = resp.json()
    assert body["photo_quality"]["uploaded"] is True
    assert "acne_prediction" in body
    assert body["acne_prediction"]["available"] is True
    assert body["acne_prediction"]["class"] in ["mild", "moderate", "severe", "very_severe"]
    assert "confidence" in body["acne_prediction"]
    assert "class_probabilities" in body["acne_prediction"]


def test_analyze_invalid_file():
    data = {
        "skin_type": "Normal",
        "age_range": "18–25",
        "concerns": "[]",
        "sensitivity": "Rarely",
        "severity": 1,
    }
    files = {"photo": ("test.txt", b"not an image", "text/plain")}
    resp = client.post("/api/analyze", data=data, files=files)
    assert resp.status_code == 415


def test_assessments_history():
    resp = client.get("/api/assessments")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
