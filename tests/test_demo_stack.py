import os
import yaml
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

def test_demo_compose_file_structure():
    compose_path = REPO_ROOT / "docker-compose.demo.yml"
    assert compose_path.exists(), "docker-compose.demo.yml must exist"
    
    with open(compose_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    services = config.get("services", {})
    expected_services = ["demo_db", "demo_redis", "demo_qdrant", "demo_backend", "demo_web"]
    for svc in expected_services:
        assert svc in services, f"Expected service '{svc}' not found in docker-compose.demo.yml"
        
    # Verify port isolation: only demo_web should publish a host port (3000)
    web_ports = services["demo_web"].get("ports", [])
    assert any("3000" in str(p) for p in web_ports), "demo_web must expose port 3000"
    
    for internal_svc in ["demo_db", "demo_redis", "demo_qdrant", "demo_backend"]:
        ports = services[internal_svc].get("ports", [])
        assert len(ports) == 0, f"Internal service '{internal_svc}' must NOT expose any host ports!"

def test_demo_env_example_and_scripts():
    env_example = REPO_ROOT / ".env.demo.example"
    assert env_example.exists(), ".env.demo.example must exist"
    
    init_script = REPO_ROOT / "scripts" / "init_demo_stack.py"
    assert init_script.exists(), "scripts/init_demo_stack.py must exist"
    
    verify_script = REPO_ROOT / "scripts" / "verify_demo_e2e.py"
    assert verify_script.exists(), "scripts/verify_demo_e2e.py must exist"

def test_demo_guide_exists():
    guide = REPO_ROOT / "DEMO_GUIDE.md"
    assert guide.exists(), "DEMO_GUIDE.md must exist"
    content = guide.read_text(encoding="utf-8")
    assert "http://localhost:3000" in content
    assert "docker-compose.demo.yml" in content
    assert "verify_demo_e2e.py" in content
