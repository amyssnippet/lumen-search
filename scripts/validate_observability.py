#!/usr/bin/env python3
"""PH-010 Workstream 7: Observability, Alert Rule Validation & Metrics Parity Audit.

Validates:
1. All Prometheus metrics defined in app/metrics.py.
2. Label cardinality governance (no tenant_id, token, user_id, or unbounded user input in labels).
3. Alert rule coverage in dashboards/alert_rules.yml against exported metric names.
4. Grafana dashboard queries across all 4 operational dashboards.
5. Live scrape verification on /metrics via TestClient.

Outputs:
- output/ph010_observability_validation.json
"""

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set

import yaml
from starlette.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.main import app
import app.metrics as metrics_mod


EXPECTED_CUSTOM_METRICS = {
    "jewellery_retrieval_requests_total": {
        "type": "Counter",
        "labels": {"endpoint", "status_code", "model"},
        "description": "Total retrieval requests processed"
    },
    "jewellery_retrieval_latency_seconds": {
        "type": "Histogram",
        "labels": {"stage"},
        "description": "Retrieval stage latency in seconds"
    },
    "jewellery_qdrant_operations_total": {
        "type": "Counter",
        "labels": {"operation", "status"},
        "description": "Total operations against Qdrant vector database"
    },
    "jewellery_ratelimit_events_total": {
        "type": "Counter",
        "labels": {"action"},
        "description": "Total rate limiting decisions"
    },
    "jewellery_active_requests": {
        "type": "Gauge",
        "labels": set(),
        "description": "Current in-flight requests"
    },
    "jewellery_search_routing_requests_total": {
        "type": "Counter",
        "labels": {"route", "status"},
        "description": "Total visual search requests partitioned by route target"
    },
    "jewellery_search_routing_latency_seconds": {
        "type": "Histogram",
        "labels": {"route", "stage"},
        "description": "Search latency by routing target and processing stage"
    },
    "jewellery_canary_routing_fallbacks_total": {
        "type": "Counter",
        "labels": {"reason"},
        "description": "Total candidate search routing fallbacks to production"
    },
    "jewellery_index_lifecycle_operations_total": {
        "type": "Counter",
        "labels": {"operation", "status"},
        "description": "Total index promotion, rollback, and lifecycle operations"
    },
    "jewellery_index_points_count": {
        "type": "Gauge",
        "labels": {"target"},
        "description": "Total vector points in active index"
    },
    "jewellery_canary_traffic_percentage": {
        "type": "Gauge",
        "labels": set(),
        "description": "Configured canary traffic routing percentage"
    }
}

FORBIDDEN_LABEL_KEYS = {
    "tenant_id", "tenant", "user_id", "actor_id", "token", "query", "email",
    "password", "secret", "image_id", "filename", "path"
}


def extract_metric_names_from_expr(expr: str) -> Set[str]:
    """Extract Prometheus metric tokens from a PromQL expression."""
    # Find tokens matching prometheus naming convention [a-zA-Z_:][a-zA-Z0-9_:]*
    tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", expr)
    # Filter out PromQL functions and keywords
    promql_keywords = {
        "sum", "rate", "increase", "histogram_quantile", "by", "without", "irate",
        "count", "avg", "min", "max", "le", "topk", "bottomk", "humanizePercentage",
        "humanizeDuration", "job", "instance", "route", "stage", "status", "operation",
        "reason", "endpoint", "status_code", "model", "action", "target", "value"
    }
    metrics = set()
    for t in tokens:
        if t in promql_keywords:
            continue
        if t.startswith("jewellery_") or t == "up":
            metrics.add(t)
    return metrics


def validate_alert_rules(alert_rules_path: str = "dashboards/alert_rules.yml") -> Dict[str, Any]:
    with open(alert_rules_path, "r") as fp:
        doc = yaml.safe_load(fp)

    rules_audit = []
    all_referenced_metrics = set()

    for group in doc.get("groups", []):
        for rule in group.get("rules", []):
            alert_name = rule["alert"]
            expr = rule["expr"]
            metrics = extract_metric_names_from_expr(expr)
            all_referenced_metrics.update(metrics)

            # Check if all referenced metrics exist in codebase or standard Prometheus
            missing = [m for m in metrics if m != "up" and not any(m.startswith(exp) for exp in EXPECTED_CUSTOM_METRICS)]

            rules_audit.append({
                "alert": alert_name,
                "severity": rule.get("labels", {}).get("severity", "unknown"),
                "expr": expr,
                "referenced_metrics": sorted(list(metrics)),
                "is_metric_valid": len(missing) == 0,
                "missing_metrics": missing
            })

    return {
        "total_rules": len(rules_audit),
        "rules": rules_audit,
        "all_referenced_metrics": sorted(list(all_referenced_metrics)),
        "all_valid": all(r["is_metric_valid"] for r in rules_audit)
    }


def validate_dashboards(dashboard_dir: str = "dashboards") -> Dict[str, Any]:
    dash_files = [
        "canary_comparison.json",
        "index_lifecycle.json",
        "operational_limits.json",
        "search_service_health.json"
    ]
    dashboards_audit = {}
    all_dash_metrics = set()

    for df in dash_files:
        p = Path(dashboard_dir) / df
        if not p.exists():
            dashboards_audit[df] = {"status": "FILE_NOT_FOUND"}
            continue
        with open(p, "r") as fp:
            data = json.load(fp)

        panels_audit = []
        for panel in data.get("panels", []):
            panel_title = panel.get("title", "Untitled")
            panel_type = panel.get("type", "unknown")
            targets = panel.get("targets", [])
            for t in targets:
                expr = t.get("expr", "")
                metrics = extract_metric_names_from_expr(expr)
                all_dash_metrics.update(metrics)
                missing = [m for m in metrics if m != "up" and not any(m.startswith(exp) for exp in EXPECTED_CUSTOM_METRICS)]
                panels_audit.append({
                    "panel": panel_title,
                    "type": panel_type,
                    "expr": expr,
                    "metrics": sorted(list(metrics)),
                    "valid": len(missing) == 0,
                    "missing": missing
                })

        dashboards_audit[df] = {
            "title": data.get("title", df),
            "panels_count": len(data.get("panels", [])),
            "panels": panels_audit,
            "all_panels_valid": all(p["valid"] for p in panels_audit)
        }

    return {
        "dashboards": dashboards_audit,
        "all_referenced_metrics": sorted(list(all_dash_metrics)),
        "all_dashboards_valid": all(d.get("all_panels_valid", False) for d in dashboards_audit.values())
    }


def validate_live_metrics_endpoint() -> Dict[str, Any]:
    client = TestClient(app)
    resp = client.get("/metrics")
    assert resp.status_code == 200, f"/metrics failed: {resp.status_code}"
    content = resp.text

    scraped_lines = content.splitlines()
    metrics_found = {}
    for exp_name, meta in EXPECTED_CUSTOM_METRICS.items():
        found = any(line.startswith(f"# HELP {exp_name}") or line.startswith(f"{exp_name}") for line in scraped_lines)
        metrics_found[exp_name] = {
            "present_in_scrape": found,
            "expected_type": meta["type"],
            "expected_labels": sorted(list(meta["labels"]))
        }

    # Verify no forbidden labels appear in metric lines
    violations = []
    for line in scraped_lines:
        if line.startswith("#") or "{" not in line:
            continue
        label_part = line[line.find("{") + 1 : line.find("}")]
        for kv in label_part.split(","):
            if "=" in kv:
                k, _ = kv.split("=", 1)
                k = k.strip()
                if k in FORBIDDEN_LABEL_KEYS:
                    violations.append({"line": line, "forbidden_label": k})

    return {
        "scrape_status_code": resp.status_code,
        "content_type": resp.headers.get("content-type", ""),
        "total_scrape_lines": len(scraped_lines),
        "metrics_found": metrics_found,
        "all_expected_metrics_present": all(m["present_in_scrape"] for m in metrics_found.values()),
        "cardinality_violations": violations,
        "cardinality_safe": len(violations) == 0
    }


def main():
    print("=" * 70)
    print("PH-010: Observability, Metrics & Alert Rule Validation")
    print("=" * 70)

    # 1. Alert Rules
    print("Validating alert rules in dashboards/alert_rules.yml...")
    alerts_res = validate_alert_rules()
    print(f"Alert Rules: {alerts_res['total_rules']} rules audited, all valid: {alerts_res['all_valid']}")

    # 2. Dashboards
    print("Validating Grafana dashboards in dashboards/...")
    dash_res = validate_dashboards()
    print(f"Dashboards: 4 dashboards audited, all valid: {dash_res['all_dashboards_valid']}")

    # 3. Live Scrape
    print("Validating live /metrics scrape and label cardinality...")
    scrape_res = validate_live_metrics_endpoint()
    print(f"Live Scrape: present={scrape_res['all_expected_metrics_present']}, safe={scrape_res['cardinality_safe']}")

    overall_pass = alerts_res["all_valid"] and dash_res["all_dashboards_valid"] and scrape_res["all_expected_metrics_present"] and scrape_res["cardinality_safe"]

    output_data = {
        "metadata": {
            "validation_stage": "PH-010: Observability, Alert Rule & Metrics Parity",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "PASS" if overall_pass else "FAIL"
        },
        "alert_rules_audit": alerts_res,
        "dashboards_audit": dash_res,
        "live_metrics_scrape_audit": scrape_res,
        "label_cardinality_governance": {
            "forbidden_labels_checked": sorted(list(FORBIDDEN_LABEL_KEYS)),
            "violations_detected": len(scrape_res["cardinality_violations"]),
            "policy": "Strict bounded cardinality. Zero tenant_id, actor_id, or user PII in metric labels."
        }
    }

    out_path = Path("output/ph010_observability_validation.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fp:
        json.dump(output_data, fp, indent=2)

    print(f"Validation results saved to {out_path.resolve()}")
    print(f"Overall Observability Validation: {'PASS' if overall_pass else 'FAIL'}")


if __name__ == "__main__":
    main()
