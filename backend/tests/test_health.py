"""Tests — System endpoints (GET / and GET /health)."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_root_returns_welcome(client: AsyncClient):
    resp = await client.get("/")
    assert resp.status_code == 200
    body = resp.json()
    assert "message" in body
    assert "SKINORA" in body["message"]
    assert "version" in body


@pytest.mark.asyncio
async def test_health_returns_healthy(client: AsyncClient):
    resp = await client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "healthy"
    assert "service" in body
    assert "version" in body


@pytest.mark.asyncio
async def test_docs_accessible(client: AsyncClient):
    """Swagger UI should be reachable."""
    resp = await client.get("/docs")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_openapi_schema(client: AsyncClient):
    """OpenAPI JSON schema should be valid."""
    resp = await client.get("/openapi.json")
    assert resp.status_code == 200
    schema = resp.json()
    assert schema["info"]["title"] == "SKINORA API"
