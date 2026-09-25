"""Tests — Authentication endpoints (register, login, /me)."""

import pytest
from httpx import AsyncClient


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_register_success(client: AsyncClient):
    resp = await client.post(
        "/api/v1/auth/register",
        json={
            "email": "alice@skinora.dev",
            "username": "alice",
            "password": "alicepassword1",
        },
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["success"] is True
    assert "access_token" in body["data"]
    assert body["data"]["token_type"] == "bearer"


@pytest.mark.asyncio
async def test_register_duplicate_email(client: AsyncClient):
    payload = {
        "email": "duplicate@skinora.dev",
        "username": "dup_user1",
        "password": "password123",
    }
    r1 = await client.post("/api/v1/auth/register", json=payload)
    assert r1.status_code == 201

    payload2 = dict(payload)
    payload2["username"] = "dup_user2"  # different username, same email
    r2 = await client.post("/api/v1/auth/register", json=payload2)
    assert r2.status_code == 409
    assert r2.json()["detail"]["code"] == "CONFLICT"


@pytest.mark.asyncio
async def test_register_duplicate_username(client: AsyncClient):
    base = {
        "email": "first@skinora.dev",
        "username": "shared_username",
        "password": "password123",
    }
    r1 = await client.post("/api/v1/auth/register", json=base)
    assert r1.status_code == 201

    base2 = dict(base)
    base2["email"] = "second@skinora.dev"
    r2 = await client.post("/api/v1/auth/register", json=base2)
    assert r2.status_code == 409


@pytest.mark.asyncio
async def test_register_weak_password(client: AsyncClient):
    resp = await client.post(
        "/api/v1/auth/register",
        json={
            "email": "weak@skinora.dev",
            "username": "weakpwuser",
            "password": "short",  # < 8 chars
        },
    )
    assert resp.status_code == 422  # Pydantic validation error


@pytest.mark.asyncio
async def test_register_invalid_email(client: AsyncClient):
    resp = await client.post(
        "/api/v1/auth/register",
        json={
            "email": "not-an-email",
            "username": "bademailuser",
            "password": "password123",
        },
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Login
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_login_success(client: AsyncClient, registered_user: dict):
    resp = await client.post(
        "/api/v1/auth/login",
        json={"email": "test@skinora.dev", "password": "securepassword123"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert "access_token" in body["data"]


@pytest.mark.asyncio
async def test_login_wrong_password(client: AsyncClient, registered_user: dict):
    resp = await client.post(
        "/api/v1/auth/login",
        json={"email": "test@skinora.dev", "password": "wrongpassword"},
    )
    assert resp.status_code == 401
    assert resp.json()["detail"]["code"] == "INVALID_CREDENTIALS"


@pytest.mark.asyncio
async def test_login_nonexistent_user(client: AsyncClient):
    resp = await client.post(
        "/api/v1/auth/login",
        json={"email": "nobody@skinora.dev", "password": "password123"},
    )
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Protected route: GET /api/v1/users/me
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_me_authenticated(client: AsyncClient, auth_headers: dict):
    resp = await client.get("/api/v1/users/me", headers=auth_headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["email"] == "test@skinora.dev"
    assert body["username"] == "testuser"
    assert "hashed_password" not in body  # Never leak the hash


@pytest.mark.asyncio
async def test_get_me_unauthenticated(client: AsyncClient):
    resp = await client.get("/api/v1/users/me")
    assert resp.status_code in (401, 403)  # HTTPBearer raises 403 or 401 depending on FastAPI version


@pytest.mark.asyncio
async def test_get_me_invalid_token(client: AsyncClient):
    resp = await client.get(
        "/api/v1/users/me",
        headers={"Authorization": "Bearer this.is.not.valid"},
    )
    assert resp.status_code == 401
