def test_register_and_login(client):
    resp = client.post("/api/auth/register", json={
        "name": "Test User", "email": "test@example.com",
        "password": "password123", "confirm_password": "password123",
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["role"] == "user"
    assert "access_token" in body

    resp2 = client.post("/api/auth/login", json={"email": "test@example.com", "password": "password123"})
    assert resp2.status_code == 200
    assert "access_token" in resp2.json()


def test_register_password_mismatch(client):
    resp = client.post("/api/auth/register", json={
        "name": "Test User", "email": "mismatch@example.com",
        "password": "password123", "confirm_password": "different123",
    })
    assert resp.status_code == 400


def test_login_wrong_password(client):
    client.post("/api/auth/register", json={
        "name": "Test User", "email": "wrongpass@example.com",
        "password": "password123", "confirm_password": "password123",
    })
    resp = client.post("/api/auth/login", json={"email": "wrongpass@example.com", "password": "nope"})
    assert resp.status_code == 401


def test_me_requires_auth(client):
    resp = client.get("/api/auth/me")
    assert resp.status_code == 401
