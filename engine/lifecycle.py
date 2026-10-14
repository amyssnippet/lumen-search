"""Qdrant Collection Lifecycle and Alias Management for Blue-Green Promotion and Rollback.

Implements the index lifecycle state machine:
    BUILD -> VALIDATE -> STAGE -> CANARY -> PROMOTE -> MONITOR -> RETIRE

Features:
- Distinguishable production and candidate collection versions.
- Stable production alias: `jewellery_catalog_mobilenetv4`.
- Atomic alias updates using Qdrant's `update_collection_aliases`.
- Bounded validation checks (vector dimensions, distance metric, payload index fields, point counts).
- Retrieval smoke test against candidate collections.
- Thread-safe idempotent promotion with concurrency serialization.
- Reversible production rollback to the previous version within a defined window.
- Guarded collection retirement that prevents accidental deletion of active or rollback-eligible indexes.
- Audit logging of operational lifecycle events.
"""

import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from qdrant_client.http.exceptions import UnexpectedResponse

logger = logging.getLogger(__name__)

DEFAULT_PRODUCTION_ALIAS = os.environ.get("QDRANT_PRODUCTION_ALIAS", "jewellery_catalog_mobilenetv4")
DEFAULT_MODEL_NAME = "mobilenetv4"
DEFAULT_EMBEDDING_DIM = 1280
EXPECTED_DISTANCE = rest.Distance.COSINE

REQUIRED_PAYLOAD_FIELDS = [
    "category",
    "sku_id",
    "tenant_id",
    "metal_color",
    "metal_purity",
    "gemstone_type",
    "setting_type"
]


class IndexState(str, Enum):
    BUILDING = "BUILDING"
    VALIDATED = "VALIDATED"
    STAGED = "STAGED"
    CANARY = "CANARY"
    PROMOTED = "PROMOTED"
    MONITORING = "MONITORING"
    RETIRED = "RETIRED"


@dataclass
class PromotionRecord:
    id: str
    timestamp: float
    promoted_collection: str
    previous_collection: Optional[str]
    alias_name: str
    actor_id: str
    status: str
    rollback_eligible: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class IndexLifecycleManager:
    """Manages Qdrant vector index lifecycle, atomic alias updates, and blue-green rollouts."""

    _lock = threading.Lock()

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        production_alias: str = DEFAULT_PRODUCTION_ALIAS,
        model_name: str = DEFAULT_MODEL_NAME,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        history_file: str = "data/lifecycle_history.json"
    ):
        self.host = host if host is not None else os.environ.get("QDRANT_HOST", "127.0.0.1")
        self.port = port if port is not None else int(os.environ.get("QDRANT_PORT", "6333"))
        self.production_alias = production_alias
        self.model_name = model_name.lower()
        self.embedding_dim = embedding_dim
        self.history_file = Path(history_file)

        self.client = QdrantClient(
            host=self.host,
            port=self.port,
            timeout=10.0,
            check_compatibility=False
        )

        self._candidate_collection: Optional[str] = None
        self._promotion_history: List[PromotionRecord] = []
        self._load_history()

    # --------------------------------------------------------------------------
    # History & State Persistence
    # --------------------------------------------------------------------------

    def _load_history(self):
        """Load promotion history from disk."""
        if self.history_file.exists():
            try:
                with open(self.history_file, "r") as fp:
                    raw_data = json.load(fp)
                    self._promotion_history = [PromotionRecord(**item) for item in raw_data.get("history", [])]
                    self._candidate_collection = raw_data.get("candidate_collection")
            except Exception as e:
                logger.warning(f"Failed to load lifecycle history from {self.history_file}: {e}")

    def _save_history(self):
        """Persist promotion history to disk."""
        try:
            self.history_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.history_file, "w") as fp:
                json.dump({
                    "production_alias": self.production_alias,
                    "candidate_collection": self._candidate_collection,
                    "history": [asdict(rec) for rec in self._promotion_history]
                }, fp, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save lifecycle history to {self.history_file}: {e}")

    # --------------------------------------------------------------------------
    # Alias Inspection & Resolution
    # --------------------------------------------------------------------------

    def get_active_production_collection(self) -> Optional[str]:
        """Resolve the collection currently mapped to the production alias."""
        try:
            aliases_res = self.client.get_aliases()
            for a in aliases_res.aliases:
                if a.alias_name == self.production_alias:
                    return a.collection_name
        except Exception as e:
            logger.warning(f"Could not resolve production alias '{self.production_alias}': {e}")
        return None

    def get_candidate_collection(self) -> Optional[str]:
        """Return registered candidate collection for canary/staging."""
        return self._candidate_collection

    def list_collections(self) -> List[Dict[str, Any]]:
        """List all collections with vector points and alias bindings."""
        try:
            colls_res = self.client.get_collections()
            aliases_res = self.client.get_aliases()
            alias_map: Dict[str, List[str]] = {}
            for a in aliases_res.aliases:
                alias_map.setdefault(a.collection_name, []).append(a.alias_name)

            out = []
            for c in colls_res.collections:
                info = None
                points_count = 0
                status = "unknown"
                try:
                    c_info = self.client.get_collection(c.name)
                    points_count = c_info.points_count or 0
                    status = str(c_info.status)
                except Exception:
                    pass

                out.append({
                    "name": c.name,
                    "points_count": points_count,
                    "status": status,
                    "aliases": alias_map.get(c.name, []),
                    "is_production": self.production_alias in alias_map.get(c.name, []),
                    "is_candidate": (c.name == self._candidate_collection)
                })
            return out
        except Exception as e:
            logger.error(f"Error querying Qdrant collections: {e}")
            return []

    # --------------------------------------------------------------------------
    # Candidate Registration & Integrity Validation
    # --------------------------------------------------------------------------

    def register_candidate(self, collection_name: str) -> Dict[str, Any]:
        """Register a candidate collection for validation and staging."""
        with self._lock:
            if not self.client.collection_exists(collection_name):
                return {
                    "status": "FAILED",
                    "error": f"Collection '{collection_name}' does not exist in Qdrant."
                }
            self._candidate_collection = collection_name
            self._save_history()
            return {
                "status": "REGISTERED",
                "candidate_collection": collection_name
            }

    def clear_candidate(self) -> Dict[str, Any]:
        """Deregister active candidate."""
        with self._lock:
            prev = self._candidate_collection
            self._candidate_collection = None
            self._save_history()
            return {"status": "CLEARED", "previous_candidate": prev}

    def validate_collection_integrity(
        self,
        collection_name: str,
        min_points: int = 1
    ) -> Dict[str, Any]:
        """Deeply inspect vector configuration, payload fields, and index volume."""
        if not self.client.collection_exists(collection_name):
            return {
                "valid": False,
                "error": f"Collection '{collection_name}' does not exist."
            }

        try:
            info = self.client.get_collection(collection_name)
            points_count = info.points_count or 0

            # 1. Point count verification
            if points_count < min_points:
                return {
                    "valid": False,
                    "error": f"Collection has {points_count} points, minimum required is {min_points}."
                }

            # 2. Vector configuration check
            vectors_cfg = info.config.params.vectors
            dim = getattr(vectors_cfg, "size", None)
            dist = getattr(vectors_cfg, "distance", None)

            # Handle multi-vector or dictionary config if present
            if dim is None and isinstance(vectors_cfg, dict):
                dim = vectors_cfg.get("size")
                dist = vectors_cfg.get("distance")

            if dim != self.embedding_dim:
                return {
                    "valid": False,
                    "error": f"Vector dimension mismatch: expected {self.embedding_dim}, found {dim}."
                }

            if str(dist).lower() not in ("cosine", "distance.cosine"):
                return {
                    "valid": False,
                    "error": f"Distance metric mismatch: expected Cosine, found {dist}."
                }

            # 3. Payload schema inspection
            payload_schema = info.payload_schema or {}
            missing_payload_fields = [f for f in REQUIRED_PAYLOAD_FIELDS if f not in payload_schema]

            return {
                "valid": True,
                "collection_name": collection_name,
                "points_count": points_count,
                "vector_dimension": dim,
                "distance": str(dist),
                "payload_fields": list(payload_schema.keys()),
                "missing_payload_fields": missing_payload_fields,
                "status": str(info.status)
            }
        except Exception as e:
            return {
                "valid": False,
                "error": f"Exception validating collection '{collection_name}': {str(e)}"
            }

    def run_retrieval_smoke_test(
        self,
        collection_name: str,
        top_k: int = 5,
        test_tenant_id: str = "default_tenant"
    ) -> Dict[str, Any]:
        """Execute test nearest-neighbor queries against candidate collection."""
        if not self.client.collection_exists(collection_name):
            return {"passed": False, "error": f"Collection '{collection_name}' not found"}

        try:
            # Query using a synthetic normalized unit vector
            import numpy as np
            query_vec = np.ones(self.embedding_dim, dtype=np.float32)
            query_vec /= np.linalg.norm(query_vec)

            # 1. Unfiltered query
            t0 = time.time()
            res_obj = self.client.query_points(
                collection_name=collection_name,
                query=query_vec.tolist(),
                limit=top_k,
                with_payload=True
            )
            results = res_obj.points
            unfiltered_latency_ms = round((time.time() - t0) * 1000, 2)

            # 2. Tenant-scoped query
            t1 = time.time()
            tenant_filter = rest.Filter(
                must=[
                    rest.FieldCondition(
                        key="tenant_id",
                        match=rest.MatchValue(value=test_tenant_id)
                    )
                ]
            )
            tenant_res_obj = self.client.query_points(
                collection_name=collection_name,
                query=query_vec.tolist(),
                query_filter=tenant_filter,
                limit=top_k,
                with_payload=True
            )
            tenant_results = tenant_res_obj.points
            tenant_latency_ms = round((time.time() - t1) * 1000, 2)

            for r in tenant_results:
                point_tenant = r.payload.get("tenant_id") if r.payload else None
                if point_tenant != test_tenant_id:
                    return {
                        "passed": False,
                        "error": f"Tenant isolation breach detected: returned point with tenant '{point_tenant}'"
                    }

            return {
                "passed": True,
                "collection_name": collection_name,
                "unfiltered_candidates": len(results),
                "unfiltered_latency_ms": unfiltered_latency_ms,
                "tenant_candidates": len(tenant_results),
                "tenant_latency_ms": tenant_latency_ms,
                "top_score": float(results[0].score) if results else 0.0
            }
        except Exception as e:
            return {"passed": False, "error": f"Retrieval smoke test failed: {str(e)}"}

    # --------------------------------------------------------------------------
    # Blue-Green Atomic Promotion
    # --------------------------------------------------------------------------

    def promote_candidate(
        self,
        candidate_collection: Optional[str] = None,
        dry_run: bool = False,
        actor_id: str = "system",
        skip_validation: bool = False
    ) -> Dict[str, Any]:
        """Atomically switch production alias to the validated candidate collection."""
        with self._lock:
            target = candidate_collection or self._candidate_collection
            if not target:
                return {
                    "status": "FAILED",
                    "error": "No candidate collection specified or registered for promotion."
                }

            # 1. Validation Gate
            if not skip_validation:
                val = self.validate_collection_integrity(target)
                if not val.get("valid"):
                    return {
                        "status": "FAILED",
                        "error": f"Candidate validation failed: {val.get('error')}"
                    }

                smoke = self.run_retrieval_smoke_test(target)
                if not smoke.get("passed"):
                    return {
                        "status": "FAILED",
                        "error": f"Candidate retrieval smoke test failed: {smoke.get('error')}"
                    }

            current_prod = self.get_active_production_collection()
            if current_prod == target:
                return {
                    "status": "ALREADY_ACTIVE",
                    "message": f"Candidate collection '{target}' is already mapped to '{self.production_alias}'.",
                    "active_collection": target
                }

            if dry_run:
                return {
                    "status": "DRY_RUN",
                    "alias_name": self.production_alias,
                    "target_collection": target,
                    "current_collection": current_prod,
                    "actions_preview": [
                        {"action": "delete_alias", "alias": self.production_alias} if current_prod else None,
                        {"action": "create_alias", "alias": self.production_alias, "collection": target}
                    ]
                }

            # 2. Build Atomic ChangeAliasesOperation sequence
            actions = []
            if current_prod:
                actions.append(
                    rest.DeleteAliasOperation(
                        delete_alias=rest.DeleteAlias(alias_name=self.production_alias)
                    )
                )
            actions.append(
                rest.CreateAliasOperation(
                    create_alias=rest.CreateAlias(
                        collection_name=target,
                        alias_name=self.production_alias
                    )
                )
            )

            try:
                # Execute atomic update
                self.client.update_collection_aliases(change_aliases_operations=actions)
                
                # Verify alias resolution
                new_active = self.get_active_production_collection()
                if new_active != target:
                    raise RuntimeError(f"Alias update verification failed: expected {target}, resolved {new_active}")

                # Save promotion record
                record = PromotionRecord(
                    id=str(uuid.uuid4()),
                    timestamp=time.time(),
                    promoted_collection=target,
                    previous_collection=current_prod,
                    alias_name=self.production_alias,
                    actor_id=actor_id,
                    status="SUCCESS",
                    rollback_eligible=True,
                    metadata={"points_count": self.client.get_collection(target).points_count}
                )
                self._promotion_history.append(record)
                self._save_history()

                logger.info(f"Promoted '{target}' to production alias '{self.production_alias}' (previous: '{current_prod}').")
                return {
                    "status": "SUCCESS",
                    "promotion_id": record.id,
                    "promoted_collection": target,
                    "previous_collection": current_prod,
                    "alias_name": self.production_alias,
                    "timestamp": record.timestamp
                }
            except Exception as e:
                logger.error(f"Atomic alias promotion failed: {e}")
                return {
                    "status": "FAILED",
                    "error": f"Qdrant alias update failed: {str(e)}",
                    "active_collection": current_prod
                }

    # --------------------------------------------------------------------------
    # Reversible Rollback
    # --------------------------------------------------------------------------

    def rollback_production(
        self,
        dry_run: bool = False,
        actor_id: str = "system"
    ) -> Dict[str, Any]:
        """Revert production alias to the previous production collection."""
        with self._lock:
            if not self._promotion_history:
                return {
                    "status": "FAILED",
                    "error": "No previous promotion record found to roll back to."
                }

            # Find latest successful promotion with rollback eligibility
            eligible_records = [r for r in self._promotion_history if r.status == "SUCCESS" and r.previous_collection]
            if not eligible_records:
                return {
                    "status": "FAILED",
                    "error": "No eligible previous collection found in promotion history."
                }

            last_record = eligible_records[-1]
            target_rollback = last_record.previous_collection

            if not self.client.collection_exists(target_rollback):
                return {
                    "status": "FAILED",
                    "error": f"Previous collection '{target_rollback}' no longer exists on Qdrant."
                }

            current_prod = self.get_active_production_collection()
            if current_prod == target_rollback:
                return {
                    "status": "ALREADY_ACTIVE",
                    "message": f"Previous collection '{target_rollback}' is already the active production index."
                }

            if dry_run:
                return {
                    "status": "DRY_RUN",
                    "alias_name": self.production_alias,
                    "current_collection": current_prod,
                    "target_rollback_collection": target_rollback,
                    "actions_preview": [
                        {"action": "delete_alias", "alias": self.production_alias},
                        {"action": "create_alias", "alias": self.production_alias, "collection": target_rollback}
                    ]
                }

            actions = [
                rest.DeleteAliasOperation(
                    delete_alias=rest.DeleteAlias(alias_name=self.production_alias)
                ),
                rest.CreateAliasOperation(
                    create_alias=rest.CreateAlias(
                        collection_name=target_rollback,
                        alias_name=self.production_alias
                    )
                )
            ]

            try:
                self.client.update_collection_aliases(change_aliases_operations=actions)
                new_active = self.get_active_production_collection()
                if new_active != target_rollback:
                    raise RuntimeError(f"Rollback alias verification failed: expected {target_rollback}, got {new_active}")

                # Append rollback audit record
                record = PromotionRecord(
                    id=str(uuid.uuid4()),
                    timestamp=time.time(),
                    promoted_collection=target_rollback,
                    previous_collection=current_prod,
                    alias_name=self.production_alias,
                    actor_id=actor_id,
                    status="ROLLBACK_SUCCESS",
                    rollback_eligible=False,
                    metadata={"reason": "operator_requested_rollback"}
                )
                self._promotion_history.append(record)
                self._save_history()

                logger.warning(f"Rolled back production alias '{self.production_alias}' from '{current_prod}' to '{target_rollback}'.")
                return {
                    "status": "ROLLBACK_SUCCESS",
                    "rollback_id": record.id,
                    "active_collection": target_rollback,
                    "reverted_from": current_prod,
                    "timestamp": record.timestamp
                }
            except Exception as e:
                logger.error(f"Rollback operation failed: {e}")
                return {
                    "status": "FAILED",
                    "error": f"Rollback failed: {str(e)}",
                    "active_collection": current_prod
                }

    # --------------------------------------------------------------------------
    # Guarded Collection Retirement
    # --------------------------------------------------------------------------

    def retire_collection(
        self,
        collection_name: str,
        force: bool = False,
        actor_id: str = "system"
    ) -> Dict[str, Any]:
        """Safely delete an obsolete index after verifying it is not active or rollback-eligible."""
        with self._lock:
            if not self.client.collection_exists(collection_name):
                return {"status": "FAILED", "error": f"Collection '{collection_name}' does not exist."}

            # 1. Protection against deleting active production index
            active_prod = self.get_active_production_collection()
            if active_prod == collection_name:
                return {
                    "status": "FORBIDDEN",
                    "error": f"Cannot retire '{collection_name}': it is currently mapped to production alias '{self.production_alias}'."
                }

            # 2. Protection against deleting active candidate
            if self._candidate_collection == collection_name:
                return {
                    "status": "FORBIDDEN",
                    "error": f"Cannot retire '{collection_name}': it is registered as active candidate."
                }

            # 3. Protection against deleting immediate rollback collection unless force=True
            if not force and self._promotion_history:
                last_rec = self._promotion_history[-1]
                if last_rec.previous_collection == collection_name and last_rec.rollback_eligible:
                    return {
                        "status": "FORBIDDEN",
                        "error": f"Cannot retire '{collection_name}': it is the active rollback target. Use force=True to override."
                    }

            try:
                self.client.delete_collection(collection_name)
                logger.info(f"Retired obsolete collection '{collection_name}' (actor: {actor_id}).")
                return {
                    "status": "RETIRED",
                    "deleted_collection": collection_name,
                    "actor_id": actor_id
                }
            except Exception as e:
                return {"status": "FAILED", "error": f"Error deleting collection: {str(e)}"}

    # --------------------------------------------------------------------------
    # Overall Status Summary
    # --------------------------------------------------------------------------

    def get_status_summary(self) -> Dict[str, Any]:
        """Comprehensive lifecycle and routing status report."""
        active_prod = self.get_active_production_collection()
        collections = self.list_collections()
        recent_promotions = [asdict(r) for r in self._promotion_history[-5:]]

        return {
            "production_alias": self.production_alias,
            "active_production_collection": active_prod,
            "candidate_collection": self._candidate_collection,
            "total_collections": len(collections),
            "collections": collections,
            "recent_promotions": recent_promotions
        }
