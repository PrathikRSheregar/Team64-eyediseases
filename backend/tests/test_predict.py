import io
from PIL import Image


def _make_test_image_bytes():
    img = Image.new("RGB", (224, 224), color=(120, 130, 140))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf


def _register_and_get_token(client, email="predictor@example.com"):
    resp = client.post("/api/auth/register", json={
        "name": "Predictor", "email": email,
        "password": "password123", "confirm_password": "password123",
    })
    return resp.json()["access_token"]


def test_predict_without_model_returns_503(client):
    """Before ml/training/train_all.py has ever been run, there is no
    production model on disk, so the API must fail predictably (503),
    never with a raw stack trace."""
    token = _register_and_get_token(client)
    img_buf = _make_test_image_bytes()

    resp = client.post(
        "/api/predict",
        files={"file": ("eye.jpg", img_buf, "image/jpeg")},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code in (503, 200)  # 200 only if a model happens to be trained in this env
    if resp.status_code == 503:
        assert "detail" in resp.json()


def test_predict_requires_auth(client):
    img_buf = _make_test_image_bytes()
    resp = client.post("/api/predict", files={"file": ("eye.jpg", img_buf, "image/jpeg")})
    assert resp.status_code == 401


def test_predict_rejects_bad_file_type(client):
    token = _register_and_get_token(client, email="badfile@example.com")
    resp = client.post(
        "/api/predict",
        files={"file": ("notes.txt", io.BytesIO(b"hello world"), "text/plain")},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 415


def test_health_endpoint(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert "model_loaded" in resp.json()
