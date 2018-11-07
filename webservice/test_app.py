from app import app


def test_response():
    test_app = app.test_client()
    resp = test_app.get('/').data.decode('utf-8')
    correct_resp = 'z twarzy podobny zupelnie do nikogo'
    assert resp == correct_resp
