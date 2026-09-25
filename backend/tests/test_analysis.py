"""Tests — Analysis endpoints (upload, history, single analysis, ML inference)."""

import pytest
from httpx import AsyncClient

from tests.conftest import make_jpeg_bytes, make_png_bytes
from app.ml.inference_wrapper import BackendInferenceService


# ---------------------------------------------------------------------------
# Image Upload
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upload_valid_jpeg(client: AsyncClient, auth_headers: dict):
    resp = await client.post(
        "/api/v1/analysis/upload",
        headers=auth_headers,
        files={"file": ("face.jpg", make_jpeg_bytes(), "image/jpeg")},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["success"] is True
    assert "analysis_id" in body
    assert body["filename"].endswith(".jpg")
    assert "status" in body


@pytest.mark.asyncio
async def test_upload_valid_png(client: AsyncClient, auth_headers: dict):
    resp = await client.post(
        "/api/v1/analysis/upload",
        headers=auth_headers,
        files={"file": ("skin.png", make_png_bytes(), "image/png")},
    )
    assert resp.status_code == 201
    assert resp.json()["success"] is True


@pytest.mark.asyncio
async def test_upload_unsupported_file_type(client: AsyncClient, auth_headers: dict):
    resp = await client.post(
        "/api/v1/analysis/upload",
        headers=auth_headers,
        files={"file": ("notes.txt", b"hello world", "text/plain")},
    )
    assert resp.status_code in (400, 415)


@pytest.mark.asyncio
async def test_upload_corrupt_image(client: AsyncClient, auth_headers: dict):
    resp = await client.post(
        "/api/v1/analysis/upload",
        headers=auth_headers,
        files={"file": ("fake.jpg", b"this is not an image", "image/jpeg")},
    )
    assert resp.status_code in (400, 415)


@pytest.mark.asyncio
async def test_upload_requires_auth(client: AsyncClient):
    resp = await client.post(
        "/api/v1/analysis/upload",
        files={"file": ("face.jpg", make_jpeg_bytes(), "image/jpeg")},
    )
    assert resp.status_code in (401, 403)


# ---------------------------------------------------------------------------
# ML Inference integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upload_returns_completed_status_when_model_loaded(
    client: AsyncClient, auth_headers: dict
):
    """When the model is loaded, upload should return status=completed with a real prediction."""
    if not BackendInferenceService.is_available():
        pytest.skip("ML model not loaded — skipping inference test")

    upload = await client.post(
        "/api/v1/analysis/upload",
        headers=auth_headers,
        files={"file": ("face.jpg", make_jpeg_bytes(), "image/jpeg")},
    )
    assert upload.status_code == 201
    body = upload.json()
    assert body["status"] == "completed"

    # Retrieve the analysis and verify observation was created
    analysis_id = body["analysis_id"]
    detail = await client.get(f"/api/v1/analysis/{analysis_id}", headers=auth_headers)
    assert detail.status_code == 200
    detail_body = detail.json()
    assert detail_body["status"] == "completed"
    assert len(detail_body["observations"]) == 1

    obs = detail_body["observations"][0]
    assert obs["observation_type"] == "acne"
    assert obs["value"] is not None
    assert obs["confidence"] is not None
    assert 0.0 <= obs["confidence"] <= 1.0


@pytest.mark.asyncio
async def test_observation_has_valid_acne_level(
    client: AsyncClient, auth_headers: dict
):
    """The observation value should be one of the four known class names."""
    if not BackendInferenceService.is_available():
        pytest.skip("ML model not loaded — skipping inference test")

    valid_classes = {
        "Level 0 (Clear)", "Level 1 (Mild)",
        "Level 2 (Moderate)", "Level 3 (Severe)",
    }

    upload = await client.post(
        "/api/v1/analysis/upload",
        headers=auth_headers,
        files={"file": ("face.jpg", make_jpeg_bytes(), "image/jpeg")},
    )
    assert upload.status_code == 201
    analysis_id = upload.json()["analysis_id"]

    obs_resp = await client.get(
        f"/api/v1/observations/{analysis_id}", headers=auth_headers
    )
    assert obs_resp.status_code == 200
    observations = obs_resp.json()
    assert len(observations) == 1
    assert observations[0]["value"] in valid_classes


@pytest.mark.asyncio
async def test_ml_status_endpoint(client: AsyncClient):
    """GET /api/v1/ml/status should return model_loaded=true when model is available."""
    resp = await client.get("/api/v1/ml/status")
    assert resp.status_code == 200
    body = resp.json()
    assert "model_loaded" in body
    assert "model_path" in body

    if BackendInferenceService.is_available():
        assert body["model_loaded"] is True


# ---------------------------------------------------------------------------
# Analysis history
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_analysis_history(client: AsyncClient, auth_headers: dict):
    for _ in range(2):
        await client.post(
            "/api/v1/analysis/upload",
            headers=auth_headers,
            files={"file": ("face.jpg", make_jpeg_bytes(), "image/jpeg")},
        )

    resp = await client.get("/api/v1/analysis/history", headers=auth_headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert isinstance(body["analyses"], list)
    assert body["total"] >= 2


@pytest.mark.asyncio
async def test_get_analysis_not_found(client: AsyncClient, auth_headers: dict):
    resp = await client.get(
        "/api/v1/analysis/00000000-0000-0000-0000-000000000000",
        headers=auth_headers,
    )
    assert resp.status_code == 404
    assert resp.json()["detail"]["code"] == "NOT_FOUND"


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recommendations_no_severity(client: AsyncClient, auth_headers: dict):
    resp = await client.get("/api/v1/recommendations", headers=auth_headers)
    assert resp.status_code == 200
    body = resp.json()
    assert "guidance" in body
    assert "disclaimer" in body


@pytest.mark.asyncio
async def test_recommendations_with_severity(client: AsyncClient, auth_headers: dict):
    for level in range(4):
        resp = await client.get(
            f"/api/v1/recommendations?acne_severity={level}", headers=auth_headers
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "guidance" in body
        assert "disclaimer" in body


@pytest.mark.asyncio
async def test_recommendations_invalid_severity(client: AsyncClient, auth_headers: dict):
    resp = await client.get(
        "/api/v1/recommendations?acne_severity=99", headers=auth_headers
    )
    assert resp.status_code == 422
