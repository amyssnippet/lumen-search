"""Prometheus observability metrics collector for the Jewellery Visual Retrieval Engine."""

import time
from typing import Optional
from fastapi import Response
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
    REGISTRY,
)

# Request counters (Bounded cardinality: no tenant_id, user_id, or token in labels!)
RETRIEVAL_REQUESTS_TOTAL = Counter(
    "jewellery_retrieval_requests_total",
    "Total retrieval requests processed",
    ["endpoint", "status_code", "model"]
)

# Latency histogram by pipeline stage
RETRIEVAL_LATENCY_SECONDS = Histogram(
    "jewellery_retrieval_latency_seconds",
    "Retrieval stage latency in seconds",
    ["stage"],
    buckets=[0.005, 0.010, 0.025, 0.050, 0.075, 0.100, 0.200, 0.500, 1.0, 2.0]
)

# Qdrant operations counter
QDRANT_OPERATIONS_TOTAL = Counter(
    "jewellery_qdrant_operations_total",
    "Total operations against Qdrant vector database",
    ["operation", "status"]
)

# Rate limiting events counter
RATELIMIT_EVENTS_TOTAL = Counter(
    "jewellery_ratelimit_events_total",
    "Total rate limiting decisions",
    ["action"]  # allowed, blocked, degraded_fail_secure
)

# In-flight requests gauge
ACTIVE_REQUESTS_GAUGE = Gauge(
    "jewellery_active_requests",
    "Current in-flight requests"
)

# Canary search routing counters (Bounded cardinality: route in [prod, canary], status in [success, error])
SEARCH_ROUTING_REQUESTS_TOTAL = Counter(
    "jewellery_search_routing_requests_total",
    "Total visual search requests partitioned by route target",
    ["route", "status"]
)

# Search latency by routing path and stage
SEARCH_ROUTING_LATENCY_SECONDS = Histogram(
    "jewellery_search_routing_latency_seconds",
    "Search latency by routing target and processing stage",
    ["route", "stage"],
    buckets=[0.005, 0.010, 0.025, 0.050, 0.075, 0.100, 0.200, 0.500, 1.0, 2.0]
)

# Canary routing fallbacks counter
CANARY_ROUTING_FALLBACKS_TOTAL = Counter(
    "jewellery_canary_routing_fallbacks_total",
    "Total candidate search routing fallbacks to production",
    ["reason"]
)

# Index lifecycle operations counter
INDEX_LIFECYCLE_OPS_TOTAL = Counter(
    "jewellery_index_lifecycle_operations_total",
    "Total index promotion, rollback, and lifecycle operations",
    ["operation", "status"]
)

# Vector points gauge for active production and candidate
INDEX_POINTS_GAUGE = Gauge(
    "jewellery_index_points_count",
    "Total vector points in active index",
    ["target"]
)

# Canary configured percentage gauge
CANARY_PERCENTAGE_GAUGE = Gauge(
    "jewellery_canary_traffic_percentage",
    "Configured canary traffic routing percentage (0 to 100)"
)

# Incident simulation drill gauge
INCIDENT_SIMULATION_GAUGE = Gauge(
    "jewellery_incident_simulation_active",
    "Flag indicating active operational incident simulation drill (0 or 1)"
)
INCIDENT_SIMULATION_GAUGE.set(0)

# Automated backup metrics (PH-012)
BACKUP_FAILURES_TOTAL = Counter(
    "jewellery_backup_failures_total",
    "Total automated database and index backup failures",
    ["backup_type"]
)
BACKUP_SUCCESS_TOTAL = Counter(
    "jewellery_backup_success_total",
    "Total automated database and index backup successes",
    ["backup_type"]
)
LAST_BACKUP_TIMESTAMP_GAUGE = Gauge(
    "jewellery_last_backup_timestamp_seconds",
    "Unix timestamp of most recent successful backup"
)
LAST_BACKUP_TIMESTAMP_GAUGE.set(time.time())

# Edge TLS certificate expiration metric (PH-012)
TLS_CERT_EXPIRY_DAYS_GAUGE = Gauge(
    "jewellery_tls_cert_expiry_days",
    "Estimated days remaining before edge TLS certificate expiration"
)
TLS_CERT_EXPIRY_DAYS_GAUGE.set(365.0)


def set_incident_simulation(active: bool):
    """Set incident simulation drill flag."""
    INCIDENT_SIMULATION_GAUGE.set(1 if active else 0)


def record_request_metric(endpoint: str, status_code: int, model: str = "mobilenetv4"):
    """Increment request counter with bounded labels."""
    RETRIEVAL_REQUESTS_TOTAL.labels(
        endpoint=endpoint,
        status_code=str(status_code),
        model=model.lower()
    ).inc()


def record_stage_latency(stage: str, duration_sec: float):
    """Record latency in histogram."""
    RETRIEVAL_LATENCY_SECONDS.labels(stage=stage).observe(duration_sec)


def record_qdrant_op(operation: str, success: bool):
    """Record vector database operation status."""
    status_str = "success" if success else "error"
    QDRANT_OPERATIONS_TOTAL.labels(operation=operation, status=status_str).inc()


def record_ratelimit_event(action: str):
    """Record rate limit decision."""
    RATELIMIT_EVENTS_TOTAL.labels(action=action).inc()


def record_search_routing(route: str, success: bool = True):
    """Record search request routing decision (route: prod, canary)."""
    status_str = "success" if success else "error"
    SEARCH_ROUTING_REQUESTS_TOTAL.labels(route=route, status=status_str).inc()


def record_routing_stage_latency(route: str, stage: str, duration_sec: float):
    """Record latency partitioned by routing path and stage."""
    SEARCH_ROUTING_LATENCY_SECONDS.labels(route=route, stage=stage).observe(duration_sec)


def record_canary_fallback(reason: str):
    """Record fallback from canary candidate to production."""
    CANARY_ROUTING_FALLBACKS_TOTAL.labels(reason=reason).inc()


def record_lifecycle_op(operation: str, status: str):
    """Record index lifecycle event (promote, rollback, validate, retire)."""
    INDEX_LIFECYCLE_OPS_TOTAL.labels(operation=operation, status=status).inc()


def set_index_points(target: str, count: int):
    """Update point count gauge for active index target."""
    INDEX_POINTS_GAUGE.labels(target=target).set(count)


def set_canary_percentage(pct: int):
    """Update configured canary traffic percentage gauge."""
    CANARY_PERCENTAGE_GAUGE.set(pct)


def record_backup_event(backup_type: str, success: bool):
    """Record automated backup success or failure."""
    if success:
        BACKUP_SUCCESS_TOTAL.labels(backup_type=backup_type).inc()
        LAST_BACKUP_TIMESTAMP_GAUGE.set(time.time())
    else:
        BACKUP_FAILURES_TOTAL.labels(backup_type=backup_type).inc()


def set_tls_cert_expiry_days(days: float):
    """Record estimated days remaining before TLS certificate expiration."""
    TLS_CERT_EXPIRY_DAYS_GAUGE.set(days)


def get_metrics_response() -> Response:
    """Generate Prometheus plaintext exposition."""
    data = generate_latest(REGISTRY)
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)

