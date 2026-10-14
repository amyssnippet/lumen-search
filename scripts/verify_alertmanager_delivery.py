#!/usr/bin/env python3
"""Milestone PH-012: Alertmanager Receiver Verification & Dispatch Drill.

Verifies:
1. Alertmanager (port 9093) is healthy and reachable.
2. Injects a test alert for DatabaseBackupFailed and TLSCertificateExpiringSoon into Alertmanager.
3. Starts a temporary webhook receiver listening on port 9999 to catch dispatched notifications.
4. Validates end-to-end receipt of grouped alert notifications from Alertmanager.
5. Emits output/ph012_alertmanager_delivery.json.
"""

import json
import logging
import os
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Dict, List, Any
import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("verify_alertmanager_delivery")

ALERTMANAGER_URL = "http://127.0.0.1:9093"
OUTPUT_PATH = Path("output/ph013_alertmanager_delivery.json")

received_alerts: List[Dict[str, Any]] = []


class AlertSinkHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len).decode("utf-8")
        try:
            payload = json.loads(body)
            received_alerts.append(payload)
            logger.info(f"Captured webhook payload with {len(payload.get('alerts', []))} alerts")
        except Exception as e:
            logger.error(f"Failed to parse alert webhook body: {e}")

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"status":"received"}')

    def log_message(self, format, *args):
        # Silence default HTTP server logging
        return


def run_alert_drill() -> Dict[str, Any]:
    logger.info("=== Starting Alertmanager Notification Drill ===")
    
    # 1. Start disposable webhook server on 0.0.0.0:9999
    server = HTTPServer(("0.0.0.0", 9999), AlertSinkHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    logger.info("Disposable webhook server listening on 0.0.0.0:9999")

    # 2. Check Alertmanager status
    with httpx.Client(timeout=5.0) as client:
        r_status = client.get(f"{ALERTMANAGER_URL}/api/v2/status")
        if r_status.status_code != 200:
            raise RuntimeError(f"Alertmanager not healthy: status {r_status.status_code}")
        logger.info("Alertmanager API is active and healthy.")

    # 3. Post test alert payload
    now_iso = datetime.now(timezone.utc).isoformat()
    test_alerts = [
        {
            "labels": {
                "alertname": "DatabaseBackupFailed",
                "service": "jewellery-search",
                "severity": "critical",
                "environment": "staging",
                "drill_id": f"drill-{int(time.time())}"
            },
            "annotations": {
                "summary": "Automated test drill: database backup failure",
                "description": "Verification drill ensuring alert notification delivery pipeline operates correctly."
            },
            "startsAt": now_iso
        },
        {
            "labels": {
                "alertname": "TLSCertificateExpiringSoon",
                "service": "jewellery-search",
                "severity": "warning",
                "environment": "staging",
                "drill_id": f"drill-{int(time.time())}"
            },
            "annotations": {
                "summary": "Automated test drill: edge TLS certificate expiration",
                "description": "Verification drill for certificate expiry warning notification."
            },
            "startsAt": now_iso
        }
    ]

    with httpx.Client(timeout=5.0) as client:
        logger.info("Posting test alerts to Alertmanager /api/v2/alerts...")
        r_post = client.post(f"{ALERTMANAGER_URL}/api/v2/alerts", json=test_alerts)
        if r_post.status_code not in (200, 202):
            raise RuntimeError(f"Failed to post alerts to Alertmanager: {r_post.status_code} {r_post.text}")

    # 4. Wait up to 15 seconds for Alertmanager group_wait and dispatch
    logger.info("Waiting for Alertmanager to group and dispatch to webhook...")
    t0 = time.time()
    while time.time() - t0 < 15.0:
        if received_alerts:
            break
        time.sleep(1.0)

    server.shutdown()
    logger.info(f"Webhook drill completed. Received {len(received_alerts)} webhook deliveries.")

    delivery_successful = len(received_alerts) > 0
    delivered_alert_names = []
    if delivery_successful:
        for batch in received_alerts:
            for al in batch.get("alerts", []):
                delivered_alert_names.append(al.get("labels", {}).get("alertname"))

    report = {
        "milestone": "PH-013",
        "drill_name": "Continuous Monitoring Alertmanager Delivery Drill",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "PASS" if delivery_successful else "FAIL",
        "alertmanager_url": ALERTMANAGER_URL,
        "webhook_receiver": "http://host.docker.internal:9999/alert-sink",
        "injected_alerts_count": len(test_alerts),
        "received_webhooks_count": len(received_alerts),
        "delivered_alert_names": delivered_alert_names,
        "sample_webhook_payload": received_alerts[0] if received_alerts else None,
        "evaluation": {
            "alertmanager_healthy": True,
            "alerts_accepted_by_api": r_post.status_code in (200, 202),
            "webhook_notification_received": delivery_successful,
            "rules_tested": ["DatabaseBackupFailed", "TLSCertificateExpiringSoon"]
        }
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)

    logger.info(f"Drill report saved to {OUTPUT_PATH}. Status: {report['status']}")
    return report


if __name__ == "__main__":
    rep = run_alert_drill()
    if rep["status"] != "PASS":
        sys.exit(1)
