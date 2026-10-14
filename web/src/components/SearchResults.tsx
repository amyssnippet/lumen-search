"use client";

import React, { useState } from "react";
import { SearchResponse, SearchResult } from "@/lib/types";
import { ResultCard } from "./ResultCard";
import { ImageDetailModal } from "./ImageDetailModal";
import { Sparkles, ImageOff, AlertTriangle } from "lucide-react";

interface SearchResultsProps {
  response: SearchResponse | null;
  queryPreviewUrl?: string | null;
  onSearchWithItem?: (imageUrl: string) => void;
  isLoading?: boolean;
}

export function SearchResults({
  response,
  queryPreviewUrl,
  onSearchWithItem,
  isLoading,
}: SearchResultsProps) {
  const [selectedResult, setSelectedResult] = useState<SearchResult | null>(null);

  if (isLoading) {
    return (
      <div className="bg-white rounded-2xl border border-charcoal-200 p-12 text-center shadow-subtle space-y-4">
        <div className="w-12 h-12 rounded-full border-4 border-gold border-t-transparent animate-spin mx-auto" />
        <div>
          <h3 className="text-sm font-semibold text-charcoal-900">
            Vectorizing & Performing Nearest Neighbor Retrieval...
          </h3>
          <p className="text-xs text-charcoal-500 mt-1 font-mono">
            MobileNetV4 FP32 ONNX • Qdrant HNSW Collection
          </p>
        </div>
      </div>
    );
  }

  if (!response) {
    return (
      <div className="bg-white rounded-2xl border border-charcoal-200 p-12 text-center shadow-subtle space-y-3">
        <div className="w-12 h-12 rounded-2xl bg-ivory-100 border border-charcoal-200 flex items-center justify-center text-charcoal-400 mx-auto">
          <Sparkles className="w-6 h-6 text-gold" />
        </div>
        <h3 className="text-base font-serif font-semibold text-charcoal-900">
          Ready for Visual Discovery
        </h3>
        <p className="text-xs text-charcoal-500 max-w-sm mx-auto">
          Upload a photograph above or select a preset sample to search through the indexed catalogue using sub-50ms vector similarity.
        </p>
      </div>
    );
  }

  if (!response.results || response.results.length === 0) {
    return (
      <div className="bg-white rounded-2xl border border-charcoal-200 p-12 text-center shadow-subtle space-y-3">
        <div className="w-12 h-12 rounded-2xl bg-amber-50 border border-amber-200 flex items-center justify-center text-amber-600 mx-auto">
          <ImageOff className="w-6 h-6" />
        </div>
        <h3 className="text-base font-semibold text-charcoal-900">
          No candidates met the threshold
        </h3>
        <p className="text-xs text-charcoal-500 max-w-sm mx-auto">
          No visual matches were found for the selected category filter or similarity criteria. Try switching category to &quot;All Categories&quot; or toggling &quot;Visual Similarity&quot; mode.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-sm font-semibold text-charcoal-900 font-serif">
            Retrieved Candidates ({response.results.length})
          </h2>
          <p className="text-[11px] text-charcoal-500 font-mono">
            Query ID: {response.query_id.slice(0, 16)} • Model: {response.model_name}
          </p>
        </div>
      </div>

      {/* Explicit No-Confident-Match Diagnostic Banner */}
      {response.decision === "NO_CONFIDENT_MATCH" && (
        <div className="p-3.5 bg-amber-50/90 border border-amber-200/90 rounded-xl text-xs text-amber-950 flex items-start space-x-2.5">
          <AlertTriangle className="w-4 h-4 text-amber-600 flex-shrink-0 mt-0.5" />
          <div className="space-y-0.5">
            <p className="font-semibold text-amber-900">No Confident Exact Match Found</p>
            <p className="text-amber-800">
              The query image does not match any indexed physical SKU above the verified threshold (≥88.0%). The candidates below are low-similarity nearest neighbours displayed for style exploration only.
            </p>
          </div>
        </div>
      )}

      {/* Grid of Results */}
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
        {response.results.map((result, idx) => (
          <ResultCard
            key={`${result.sku}-${result.filename}-${idx}`}
            result={result}
            rank={idx + 1}
            onInspect={(r) => setSelectedResult(r)}
            onSearchWithItem={onSearchWithItem}
          />
        ))}
      </div>

      {/* Detail Inspection Modal */}
      <ImageDetailModal
        isOpen={selectedResult !== null}
        onClose={() => setSelectedResult(null)}
        result={selectedResult}
        queryPreviewUrl={queryPreviewUrl}
        onSearchSimilar={onSearchWithItem}
      />
    </div>
  );
}
