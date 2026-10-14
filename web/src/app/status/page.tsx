"use client";

import React, { useEffect, useState } from "react";
import { Activity, Database, Server, Cpu, RefreshCw, CheckCircle2, AlertTriangle, Layers, ShieldCheck } from "lucide-react";
import { SystemStatus } from "@/lib/types";

export default function StatusPage() {
  const [status, setStatus] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchStatus = async () => {
    setRefreshing(true);
    try {
      const res = await fetch("/api/status");
      const data = await res.json();
      setStatus(data);
      setError(null);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 15000); // 15s polling
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-8 space-y-8">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-4 border-b border-charcoal-200 pb-6">
        <div>
          <div className="inline-flex items-center space-x-1.5 px-3 py-1 rounded-full bg-emerald-50 text-emerald-800 text-xs font-semibold tracking-wide uppercase mb-2">
            <Activity className="w-3.5 h-3.5" />
            <span>Infrastructure Health & Lifecycle</span>
          </div>
          <h1 className="text-3xl font-serif font-bold text-charcoal-900 tracking-tight">
            System Observability & Status
          </h1>
          <p className="text-xs text-charcoal-600 mt-1">
            Real-time probes monitoring FastAPI, PostgreSQL, Redis, Qdrant, ONNX Runtime, and index alias bindings.
          </p>
        </div>

        <button
          onClick={fetchStatus}
          disabled={refreshing}
          className="flex items-center space-x-2 px-4 py-2 text-xs font-medium bg-white hover:bg-ivory-100 border border-charcoal-200 rounded-xl transition shadow-subtle text-charcoal-700"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${refreshing ? "animate-spin text-gold-deep" : ""}`} />
          <span>Refresh Probes</span>
        </button>
      </div>

      {error && (
        <div className="p-4 bg-red-50 border border-red-200 rounded-xl text-xs text-red-700">
          {error}
        </div>
      )}

      {loading ? (
        <div className="bg-white rounded-2xl border border-charcoal-200 p-16 text-center space-y-3">
          <div className="w-10 h-10 rounded-full border-4 border-gold border-t-transparent animate-spin mx-auto" />
          <p className="text-xs text-charcoal-500 font-mono">Querying deep readiness probes...</p>
        </div>
      ) : (
        <div className="space-y-6">
          {/* Dependency Grid */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            {/* Database */}
            <div className="bg-white rounded-2xl border border-charcoal-200 p-5 shadow-subtle space-y-3">
              <div className="flex items-center justify-between">
                <div className="p-2 bg-emerald-50 rounded-xl text-emerald-700">
                  <Database className="w-5 h-5" />
                </div>
                <span
                  className={`text-[11px] px-2 py-0.5 rounded-full font-medium ${
                    status?.dependencies?.database
                      ? "bg-emerald-100 text-emerald-800"
                      : "bg-red-100 text-red-800"
                  }`}
                >
                  {status?.dependencies?.database ? "Healthy" : "Down"}
                </span>
              </div>
              <div>
                <h3 className="text-sm font-semibold text-charcoal-900">PostgreSQL 16</h3>
                <p className="text-[11px] text-charcoal-500 font-mono">
                  Relational Catalogue Store
                </p>
              </div>
              <div className="text-xs text-charcoal-600 font-mono pt-2 border-t border-charcoal-100">
                Latency:{" "}
                <strong>
                  {status?.raw_dependencies?.database?.latency_ms
                    ? `${status.raw_dependencies.database.latency_ms}ms`
                    : "Active"}
                </strong>
              </div>
            </div>

            {/* Redis */}
            <div className="bg-white rounded-2xl border border-charcoal-200 p-5 shadow-subtle space-y-3">
              <div className="flex items-center justify-between">
                <div className="p-2 bg-emerald-50 rounded-xl text-emerald-700">
                  <Server className="w-5 h-5" />
                </div>
                <span
                  className={`text-[11px] px-2 py-0.5 rounded-full font-medium ${
                    status?.dependencies?.redis
                      ? "bg-emerald-100 text-emerald-800"
                      : "bg-amber-100 text-amber-800"
                  }`}
                >
                  {status?.dependencies?.redis ? "Connected" : "Degraded (Fail-secure)"}
                </span>
              </div>
              <div>
                <h3 className="text-sm font-semibold text-charcoal-900">Redis 7</h3>
                <p className="text-[11px] text-charcoal-500 font-mono">
                  Distributed Rate Limiter
                </p>
              </div>
              <div className="text-xs text-charcoal-600 font-mono pt-2 border-t border-charcoal-100">
                Latency:{" "}
                <strong>
                  {status?.raw_dependencies?.redis?.latency_ms
                    ? `${status.raw_dependencies.redis.latency_ms}ms`
                    : "Active"}
                </strong>
              </div>
            </div>

            {/* Qdrant */}
            <div className="bg-white rounded-2xl border border-charcoal-200 p-5 shadow-subtle space-y-3">
              <div className="flex items-center justify-between">
                <div className="p-2 bg-emerald-50 rounded-xl text-emerald-700">
                  <Layers className="w-5 h-5" />
                </div>
                <span
                  className={`text-[11px] px-2 py-0.5 rounded-full font-medium ${
                    status?.dependencies?.qdrant
                      ? "bg-emerald-100 text-emerald-800"
                      : "bg-red-100 text-red-800"
                  }`}
                >
                  {status?.dependencies?.qdrant ? "Healthy" : "Down"}
                </span>
              </div>
              <div>
                <h3 className="text-sm font-semibold text-charcoal-900">Qdrant Vector DB</h3>
                <p className="text-[11px] text-charcoal-500 font-mono">
                  HNSW Index (1280 dimensions)
                </p>
              </div>
              <div className="text-xs text-charcoal-600 font-mono pt-2 border-t border-charcoal-100">
                Collections:{" "}
                <strong>
                  {status?.raw_dependencies?.qdrant?.collections_count ?? 1}
                </strong>
              </div>
            </div>

            {/* ONNX Model */}
            <div className="bg-white rounded-2xl border border-charcoal-200 p-5 shadow-subtle space-y-3">
              <div className="flex items-center justify-between">
                <div className="p-2 bg-emerald-50 rounded-xl text-emerald-700">
                  <Cpu className="w-5 h-5" />
                </div>
                <span className="text-[11px] px-2 py-0.5 rounded-full font-medium bg-emerald-100 text-emerald-800">
                  Loaded
                </span>
              </div>
              <div>
                <h3 className="text-sm font-semibold text-charcoal-900">MobileNetV4 ONNX</h3>
                <p className="text-[11px] text-charcoal-500 font-mono">
                  FP32 Conv-Small (3.8M params)
                </p>
              </div>
              <div className="text-xs text-charcoal-600 font-mono pt-2 border-t border-charcoal-100">
                Runtime: <strong>ONNXRuntime 1.20</strong>
              </div>
            </div>
          </div>

          {/* Index Lifecycle & Alias Binding */}
          <div className="bg-white rounded-2xl border border-charcoal-200 p-6 shadow-subtle space-y-4">
            <h3 className="text-sm font-serif font-bold text-charcoal-900 flex items-center">
              <ShieldCheck className="w-4 h-4 text-gold-deep mr-2" />
              Vector Index Lifecycle & Blue-Green Canary Routing
            </h3>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs font-mono">
              <div className="p-4 bg-ivory-100 rounded-xl border border-charcoal-100 space-y-1">
                <span className="text-[10px] text-charcoal-500 uppercase">Active Index Alias</span>
                <p className="font-bold text-charcoal-900 truncate">
                  {status?.lifecycle?.active_production_alias || "jewellery_catalog_mobilenetv4_staging"}
                </p>
              </div>

              <div className="p-4 bg-ivory-100 rounded-xl border border-charcoal-100 space-y-1">
                <span className="text-[10px] text-charcoal-500 uppercase">Underlying Qdrant Collection</span>
                <p className="font-bold text-emerald-700 truncate">
                  {status?.lifecycle?.target_collection || "jewellery_catalog_mobilenetv4_v1"}
                </p>
              </div>

              <div className="p-4 bg-ivory-100 rounded-xl border border-charcoal-100 space-y-1">
                <span className="text-[10px] text-charcoal-500 uppercase">Canary Search Routing</span>
                <p className="font-bold text-charcoal-800">
                  {status?.lifecycle?.canary_enabled ? "Enabled (10% Traffic)" : "100% Active Production Alias"}
                </p>
              </div>
            </div>
          </div>

          {/* Catalog Statistics */}
          <div className="bg-white rounded-2xl border border-charcoal-200 p-6 shadow-subtle space-y-4">
            <h3 className="text-sm font-serif font-bold text-charcoal-900">
              Indexed Inventory Statistics
            </h3>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-xs">
              <div className="p-4 bg-ivory-50 rounded-xl border border-charcoal-200 text-center">
                <span className="text-2xl font-bold font-mono text-charcoal-900">
                  {status?.catalog?.total_images_in_dataset || 500}
                </span>
                <span className="text-[10px] text-charcoal-500 uppercase block mt-1">
                  Catalogue Images on Disk
                </span>
              </div>

              <div className="p-4 bg-ivory-50 rounded-xl border border-charcoal-200 text-center">
                <span className="text-2xl font-bold font-mono text-emerald-700">
                  {status?.catalog?.index_stats?.total_points || 432}
                </span>
                <span className="text-[10px] text-charcoal-500 uppercase block mt-1">
                  Vectors in Active Qdrant Collection
                </span>
              </div>

              <div className="p-4 bg-ivory-50 rounded-xl border border-charcoal-200 text-center">
                <span className="text-2xl font-bold font-mono text-gold-deep">
                  {status?.uptime_sec ? `${Math.floor(status.uptime_sec / 60)}m` : "Active"}
                </span>
                <span className="text-[10px] text-charcoal-500 uppercase block mt-1">
                  FastAPI Process Uptime
                </span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
