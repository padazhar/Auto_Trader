from fastapi.testclient import TestClient

from core.path_manager import PathManager
from monitoring.web_app import create_app


def test_dashboard_root_uses_relative_assets_and_inline_bootstrap():
    client = TestClient(create_app(PathManager.from_project_root('.')))
    response = client.get('/')
    assert response.status_code == 200
    html = response.text
    assert 'href="styles.css?v=step22"' in html
    assert 'src="app.js?v=step62-ui-stability"' in html
    assert 'window.__AUTO_TRADER_BASE_PATH__' in html
    assert 'inline snapshot ok' in html


def test_dashboard_static_assets_and_api_are_no_store():
    client = TestClient(create_app(PathManager.from_project_root('.')))
    for path in ['/', '/app.js?v=step62-ui-stability', '/styles.css?v=step22', '/api/snapshot']:
        response = client.get(path)
        assert response.status_code == 200
        assert response.headers.get('cache-control') == 'no-store, no-cache, must-revalidate, max-age=0'
