#!/usr/bin/env python3
"""Observability & Incident Alert Simulation for Milestone PH-011.

Executes an automated operational incident drill against the live Prometheus scraper:
1. Verifies baseline alert state in Prometheus ('inactive').
2. Injects simulated operational incident condition via /api/lifecycle/simulate-incident.
3. Observes and timestamps state transitions:
   - INACTIVE -> PENDING (incident detected)
   - PENDING -> FIRING (alert duration threshold satisfied)
4. Clears simulated incident condition.
5. Observes and timestamps recovery transition:
   - FIRING -> INACTIVE (incident resolved)
6. Writes audit report to output/ph011_observability_incident_simulation.json.
"""

import json
import logging
import os
import sys
import time
from typing import Dict, Any, Optional
import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("incident_simulation")

STAGING_HTTPS_URL = "https://127.0.0.1:8443"
PROMETHEUS_URL = "http://127.0.0.1:9091"
API_KEY = "jvr_default_dev_key_2026_secret"
OUTPUT_PATH = "output/ph011_observability_incident_simulation.json"
TARGET_ALERT = "StagingIncidentSimulationActive"


def get_rule_state(client: httpx.Client, rule_name: str) -> Optional[Dict[str, Any]]:
    r = client.get(f"{PROMETHEUS_URL}/api/v1/rules")
    if r.status_code != 200:
        return None
    data = r.json()
    for g in data.get("data", {}).get("groups", []):
        for rule in g.get("rules", []):
            if rule.get("name") == rule_name:
                return rule
    return None


def run_incident_simulation():
    os.makedirs("output", exist_ok=True)
    timeline = []
    
    with httpx.Client(verify=False, timeout=10.0) as client:
        # Step 1: Baseline Check
        logger.info(f"--- Step 1: Baseline Check for Alert '{TARGET_ALERT}' ---")
        init_rule = get_rule_state(client, TARGET_ALERT)
        if not init_rule:
            raise RuntimeError(f"Alert rule '{TARGET_ALERT}' not found in Prometheus!")
        
        state_0 = init_rule.get("state")
        logger.info(f"Initial alert state: {state_0}")
        timeline.append({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "phase": "baseline",
            "state": state_0,
            "details": "Alert is quiet at start of drill"
        })

        # Step 2: Trigger Incident Simulation
        logger.info("--- Step 2: Triggering Incident Condition ---")
        t_trigger = time.time()
        r_trig = client.post(
            f"{STAGING_HTTPS_URL}/api/lifecycle/simulate-incident",
            headers={"X-API-Key": API_KEY},
            json={"active": True}
        )
        r_trig.raise_for_status()
        timeline.append({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "phase": "incident_injected",
            "state": "TRIGGERED",
            "details": "Simulation gauge set to 1 via API"
        })

        # Step 3: Wait for PENDING / FIRING
        logger.info("--- Step 3: Monitoring State Transition to FIRING ---")
        firing_observed = False
        pending_observed = False
        max_wait = 40  # 10s rule + scrape interval margin
        start_wait = time.time()

        while time.time() - start_wait < max_wait:
            time.sleep(2)
            rule = get_rule_state(client, TARGET_ALERT)
            current_state = rule.get("state")
            logger.info(f"Poll ({time.time() - start_wait:.1f}s): alert state = {current_state}")
            
            if current_state == "pending" and not pending_observed:
                pending_observed = True
                timeline.append({
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "phase": "state_transition",
                    "state": "pending",
                    "duration_from_trigger_sec": round(time.time() - t_trigger, 2)
                })

            if current_state == "firing":
                firing_observed = True
                timeline.append({
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "phase": "state_transition",
                    "state": "firing",
                    "duration_from_trigger_sec": round(time.time() - t_trigger, 2),
                    "active_alerts": rule.get("alerts", [])
                })
                break

        if not firing_observed:
            logger.error("Failed to observe FIRING state within timeout!")

        # Step 4: Clear Incident Condition
        logger.info("--- Step 4: Clearing Incident Condition (Recovery) ---")
        t_clear = time.time()
        r_clear = client.post(
            f"{STAGING_HTTPS_URL}/api/lifecycle/simulate-incident",
            headers={"X-API-Key": API_KEY},
            json={"active": False}
        )
        r_clear.raise_for_status()
        timeline.append({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "phase": "incident_cleared",
            "state": "RECOVERY_INITIATED",
            "details": "Simulation gauge reset to 0 via API"
        })

        # Step 5: Wait for Recovery (INACTIVE)
        logger.info("--- Step 5: Monitoring Recovery to INACTIVE ---")
        resolved_observed = False
        start_wait_res = time.time()
        while time.time() - start_wait_res < 30:
            time.sleep(2)
            rule = get_rule_state(client, TARGET_ALERT)
            current_state = rule.get("state")
            logger.info(f"Recovery Poll ({time.time() - start_wait_res:.1f}s): alert state = {current_state}")
            if current_state == "inactive":
                resolved_observed = True
                timeline.append({
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "phase": "state_transition",
                    "state": "inactive",
                    "duration_from_clear_sec": round(time.time() - t_clear, 2),
                    "details": "Alert successfully resolved and returned to inactive"
                })
                break

        overall_ok = firing_observed and resolved_observed

        report = {
            "milestone": "PH-011",
            "drill_name": "Observability & Alert Lifecycle Simulation Drill",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "prometheus_url": PROMETHEUS_URL,
            "target_alert": TARGET_ALERT,
            "alert_definition": {
                "expr": "jewellery_incident_simulation_active > 0",
                "for": "10s",
                "severity": "warning",
                "service": "jewellery-search"
            },
            "transition_verifications": {
                "initial_inactive_verified": state_0 == "inactive",
                "pending_transition_observed": pending_observed,
                "firing_transition_observed": firing_observed,
                "recovery_to_inactive_observed": resolved_observed
            },
            "lifecycle_timeline": timeline,
            "status": "PASS" if overall_ok else "FAIL"
        }

        with open(OUTPUT_PATH, "w") as fp:
            json.dump(report, fp, indent=2)

        logger.info(f"Incident drill completed! Status: {report['status']}, Written to {OUTPUT_PATH}")
        return overall_ok


if __name__ == "__main__":
    success = run_incident_simulation()
    sys.exit(0 if success else 1)
