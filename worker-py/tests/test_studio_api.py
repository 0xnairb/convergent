"""Integration test for Convergent Studio MVP Web GUI backend APIs."""

from __future__ import annotations

from http.server import ThreadingHTTPServer
from pathlib import Path
import sys
import threading

import httpx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "studio"))

from server import StudioWorkspace, make_studio_handler  # noqa: E402


def test_studio_mvp_gui_endpoints() -> None:
    ws = StudioWorkspace()
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_studio_handler(ws))
    url = f"http://127.0.0.1:{server.server_address[1]}"
    threading.Thread(target=server.serve_forever, daemon=True).start()

    try:
        with httpx.Client(timeout=15.0) as http:
            # 1. Serves HTML GUI
            html = http.get(f"{url}/")
            assert html.status_code == 200
            assert "Project Convergent" in html.text
            assert "Configure Ollama" in html.text

            # 2. Ollama model auto-discovery & runtime reconfiguration
            ol = http.get(f"{url}/api/ollama/models").raise_for_status().json()
            assert "models" in ol and "ollama_url" in ol
            if ol["models"]:
                chosen = ol["models"][0]["name"]
                cfg_state = http.post(f"{url}/api/config", json={"model_id": chosen}).raise_for_status().json()
                assert cfg_state["ollama"]["model_id"] == chosen

            # 3. Initial seeded state has finance & engineering at r2
            state = http.get(f"{url}/api/state").raise_for_status().json()
            assert state["spaces"]["finance"]["head_rev"] == 2
            assert state["spaces"]["engineering"]["head_rev"] == 2

            # 3. Open 0-byte overlay -> stage delta -> land at r3 -> undo at r4
            ov = http.post(f"{url}/api/overlays/open", json={"space": "finance", "owner": "agent-ui"}).raise_for_status().json()
            oid = ov["overlay_id"]
            http.post(
                f"{url}/api/overlays/stage",
                json={"overlay_id": oid, "source_uri": "policy://travel-policy.md", "text": "Cap is $1,500."},
            ).raise_for_status()
            landed = http.post(f"{url}/api/overlays/land", json={"overlay_id": oid, "message": "Bump cap"}).raise_for_status().json()
            assert landed["status"] == "landed" and landed["rev"] == 3

            undone = http.post(f"{url}/api/revisions/undo", json={"space": "finance", "rev": 3}).raise_for_status().json()
            assert undone["undo_rev"] == 4 and undone["restored"] == 1

            # 4. Toggle Bob's OpenFGA access off & back on
            toggled = http.post(f"{url}/api/principals/toggle", json={"persona": "bob"}).raise_for_status().json()
            assert toggled["principals"]["bob"]["active"] is False
            restored = http.post(f"{url}/api/principals/toggle", json={"persona": "bob"}).raise_for_status().json()
            assert restored["principals"]["bob"]["active"] is True
    finally:
        server.shutdown()
        server.server_close()
