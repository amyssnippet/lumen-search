"use client";

import React from "react";
import { Clock, Cpu, Eye, Zap, ShieldCheck, AlertTriangle } from "lucide-react";
import { SearchTiming, SearchQuality } from "@/lib/types";

interface LatencyBreakdownProps {
  timing: SearchTiming;
  clientE2E?: number;
  quality?: SearchQuality;
  hasReliableMatch: boolean;
  topSimilarity: number;
  decision?: string;
  decisionExplanation?: string;
  categoryFilterApplied?: string;
  detectedCategory?: string;
}

export function LatencyBreakdown({
  timing,
  clientE2E,
  quality,
  hasReliableMatch,
  topSimilarity,
  decision,
  decisionExplanation,
  categoryFilterApplied,
  detectedCategory,
}: LatencyBreakdownProps) {
  const topPct = (topSimilarity * 100).toFixed(1);

  const getDecisionBadge = () => {
    if (decision === "EXACT_MATCH" || topSimilarity >= 0.88) {
      return {
        classes: "bg-emerald-50 text-emerald-800 border border-emerald-200",
        label: "Exact SKU Candidate (≥88.0%)",
        isShield: true,
      };
    }
    if (decision === "VISUAL_SIMILARITY" || topSimilarity >= 0.52) {
      return {
        classes: "bg-amber-50 text-amber-800 border border-amber-200",
        label: topSimilarity >= 0.70 ? "High Similarity Design (70-88%)" : "Visual Style Candidate (52-70%)",
        isShield: false,
      };
    }
    return {
      classes: "bg-rose-50 text-rose-800 border border-rose-200",
      label: "No Confident Match (<52.0%)",
      isShield: false,
    };
  };

  const badge = getDecisionBadge();

  return (
    <div className="bg-white rounded-2xl border border-charcoal-200 p-4 shadow-subtle space-y-4">
      {/* Top Banner: Match Confidence & Latency summary */}
      <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-charcoal-100">
        <div className="flex flex-wrap items-center gap-3">
          <div
            title={decisionExplanation || "Heuristic cosine similarity score band"}
            className={`flex items-center space-x-1.5 px-3 py-1 rounded-full text-xs font-semibold ${badge.classes}`}
          >
            {badge.isShield ? (
              <ShieldCheck className="w-3.5 h-3.5 text-emerald-600" />
            ) : (
              <AlertTriangle className="w-3.5 h-3.5 text-amber-600" />
            )}
            <span>{badge.label}</span>
          </div>

          <span className="text-xs text-charcoal-600">
            Top Cosine Similarity: <strong className="text-charcoal-900 font-mono">{topPct}%</strong>
          </span>

          {categoryFilterApplied && (
            <span className="text-[11px] px-2 py-0.5 rounded bg-ivory-200 text-charcoal-700 font-mono border border-charcoal-200">
              Filter: {categoryFilterApplied}
            </span>
          )}
          {detectedCategory && !categoryFilterApplied && (
            <span className="text-[11px] px-2 py-0.5 rounded bg-ivory-100 text-charcoal-500 font-mono">
              Detected: {detectedCategory}
            </span>
          )}
        </div>

        {/* Latency headline */}
        <div className="flex items-center space-x-3 text-xs">
          <div className="flex items-center space-x-1 text-charcoal-500">
            <Clock className="w-3.5 h-3.5 text-gold-deep" />
            <span>Client E2E:</span>
            <span className="font-semibold text-charcoal-900 font-mono">
              {clientE2E ? `${clientE2E}ms` : "—"}
            </span>
          </div>
          <span className="text-charcoal-300">|</span>
          <div className="flex items-center space-x-1 text-charcoal-500">
            <Zap className="w-3.5 h-3.5 text-emerald-600" />
            <span>Engine Core:</span>
            <span className="font-semibold text-charcoal-900 font-mono">
              {timing.total.toFixed(1)}ms
            </span>
          </div>
        </div>
      </div>

      {/* Latency Stages & Bars */}
      <div>
        <div className="grid grid-cols-3 gap-2 text-center text-xs">
          <div className="p-2 bg-ivory-100 rounded-xl border border-charcoal-100">
            <span className="text-[10px] text-charcoal-500 uppercase tracking-wider block">
              1. Preprocessing
            </span>
            <span className="font-mono font-semibold text-charcoal-800">
              {timing.preprocessing.toFixed(1)}ms
            </span>
          </div>

          <div className="p-2 bg-ivory-100 rounded-xl border border-charcoal-100">
            <span className="text-[10px] text-charcoal-500 uppercase tracking-wider block">
              2. ONNX Embedding
            </span>
            <span className="font-mono font-semibold text-gold-deep">
              {timing.encoding.toFixed(1)}ms
            </span>
          </div>

          <div className="p-2 bg-ivory-100 rounded-xl border border-charcoal-100">
            <span className="text-[10px] text-charcoal-500 uppercase tracking-wider block">
              3. Qdrant HNSW Search
            </span>
            <span className="font-mono font-semibold text-charcoal-800">
              {timing.search.toFixed(1)}ms
            </span>
          </div>
        </div>

        {/* Visual progress bar */}
        <div className="w-full h-2 bg-ivory-200 rounded-full mt-2 overflow-hidden flex">
          <div
            style={{ width: `${Math.max(5, (timing.preprocessing / timing.total) * 100)}%` }}
            className="bg-charcoal-400 h-full"
            title={`Preprocessing: ${timing.preprocessing.toFixed(1)}ms`}
          />
          <div
            style={{ width: `${Math.max(10, (timing.encoding / timing.total) * 100)}%` }}
            className="bg-gold h-full"
            title={`Encoding: ${timing.encoding.toFixed(1)}ms`}
          />
          <div
            style={{ width: `${Math.max(5, (timing.search / timing.total) * 100)}%` }}
            className="bg-emerald-500 h-full"
            title={`Vector Search: ${timing.search.toFixed(1)}ms`}
          />
        </div>
      </div>

      {/* Quality Assessment Inspection */}
      {quality && (
        <div className="pt-2 border-t border-charcoal-100 flex flex-wrap items-center justify-between text-[11px] text-charcoal-600 gap-2">
          <div className="flex items-center space-x-2">
            <Eye className="w-3.5 h-3.5 text-charcoal-400" />
            <span>Image Assessment:</span>
            <span className="font-mono">
              Sharpness: <strong>{quality.sharpness_score.toFixed(1)}</strong>
            </span>
            <span className="font-mono">
              Luminance: <strong>{quality.mean_luminance.toFixed(1)}</strong>
            </span>
          </div>

          <div className="flex items-center space-x-2">
            {quality.skin_detected && (
              <span className="px-2 py-0.5 rounded bg-amber-100 text-amber-900 font-medium border border-amber-200">
                Skin / Hand Background ({((quality.skin_ratio || 0) * 100).toFixed(0)}%)
              </span>
            )}
            {quality.robust_crops && quality.robust_crops.length > 1 && (
              <span className="px-2 py-0.5 rounded bg-teal-100 text-teal-800 font-medium border border-teal-200">
                Adaptive Multi-Crop ({quality.robust_crops.length} views)
              </span>
            )}
            {quality.is_blurry && (
              <span className="px-2 py-0.5 rounded bg-amber-100 text-amber-800 font-medium">
                Blurry
              </span>
            )}
            {quality.is_underexposed && (
              <span className="px-2 py-0.5 rounded bg-amber-100 text-amber-800 font-medium">
                Underexposed
              </span>
            )}
            {!quality.is_blurry && !quality.is_underexposed && !quality.is_overexposed && !quality.skin_detected && (
              <span className="px-2 py-0.5 rounded bg-emerald-100 text-emerald-800 font-medium">
                Studio Quality Input
              </span>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
