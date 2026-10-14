"use client";

import React from "react";
import { X, CheckCircle, Diamond, ShieldCheck, Sparkles, Tag, ArrowRight } from "lucide-react";
import { SearchResult } from "@/lib/types";

interface ImageDetailModalProps {
  isOpen: boolean;
  onClose: () => void;
  result: SearchResult | null;
  queryPreviewUrl?: string | null;
  onSearchSimilar?: (imageUrl: string) => void;
}

export function ImageDetailModal({
  isOpen,
  onClose,
  result,
  queryPreviewUrl,
  onSearchSimilar,
}: ImageDetailModalProps) {
  if (!isOpen || !result) return null;

  const simPct = (result.similarity * 100).toFixed(2);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4 overflow-y-auto">
      <div className="bg-white rounded-2xl max-w-3xl w-full shadow-2xl border border-charcoal-200 overflow-hidden my-8 animate-in fade-in zoom-in-95 duration-150">
        {/* Header */}
        <div className="px-6 py-4 bg-ivory-100 border-b border-charcoal-200 flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <span className="font-serif text-base font-semibold text-charcoal-900">
              Candidate Visual Inspection
            </span>
            <span className="text-xs px-2.5 py-0.5 rounded-full bg-charcoal-200 text-charcoal-800 font-mono">
              SKU: {result.sku}
            </span>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-charcoal-400 hover:text-charcoal-700 hover:bg-charcoal-200 transition"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content Body */}
        <div className="p-6 space-y-6">
          {/* Comparison Row */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Query Image (if available) */}
            {queryPreviewUrl && (
              <div className="border border-charcoal-200 rounded-xl p-3 bg-ivory-50 flex flex-col items-center">
                <span className="text-xs font-semibold text-charcoal-600 mb-2 uppercase tracking-wider">
                  Query Photograph
                </span>
                <div className="w-full h-64 bg-white rounded-lg overflow-hidden border border-charcoal-100 flex items-center justify-center">
                  <img
                    src={queryPreviewUrl}
                    alt="Query"
                    className="w-full h-full object-contain"
                  />
                </div>
              </div>
            )}

            {/* Candidate Image */}
            <div
              className={`border border-charcoal-200 rounded-xl p-3 bg-ivory-50 flex flex-col items-center ${
                !queryPreviewUrl ? "md:col-span-2" : ""
              }`}
            >
              <div className="w-full flex items-center justify-between mb-2">
                <span className="text-xs font-semibold text-charcoal-600 uppercase tracking-wider">
                  Catalogue Match
                </span>
                <span className="text-xs font-bold text-gold-deep font-mono">
                  {simPct}% Similarity
                </span>
              </div>
              <div className="w-full h-64 bg-white rounded-lg overflow-hidden border border-charcoal-100 flex items-center justify-center">
                <img
                  src={result.image_url}
                  alt={result.sku}
                  className="w-full h-full object-contain"
                />
              </div>
            </div>
          </div>

          {/* Metadata Grid */}
          <div className="bg-ivory-100 rounded-xl p-4 border border-charcoal-200 grid grid-cols-2 sm:grid-cols-3 gap-4 text-xs">
            <div>
              <span className="text-charcoal-500 block mb-0.5">Category</span>
              <span className="font-semibold text-charcoal-900 capitalize">
                {result.category}
              </span>
            </div>

            <div>
              <span className="text-charcoal-500 block mb-0.5">Cosine Similarity</span>
              <span className="font-mono font-semibold text-emerald-700">
                {result.similarity.toFixed(4)} ({simPct}%)
              </span>
            </div>

            <div>
              <span className="text-charcoal-500 block mb-0.5">Vector Distance</span>
              <span className="font-mono font-semibold text-charcoal-800">
                {result.distance.toFixed(4)}
              </span>
            </div>

            {result.metal_color && (
              <div>
                <span className="text-charcoal-500 block mb-0.5">Metal / Alloy</span>
                <span className="font-medium text-charcoal-900">{result.metal_color}</span>
              </div>
            )}

            {result.gemstone_type && (
              <div>
                <span className="text-charcoal-500 block mb-0.5">Gemstone</span>
                <span className="font-medium text-charcoal-900">{result.gemstone_type}</span>
              </div>
            )}

            {result.setting_type && (
              <div>
                <span className="text-charcoal-500 block mb-0.5">Setting Style</span>
                <span className="font-medium text-charcoal-900">{result.setting_type}</span>
              </div>
            )}

            <div>
              <span className="text-charcoal-500 block mb-0.5">Match Type</span>
              <span className="font-medium text-charcoal-900">
                {result.exact_match_candidate ? (
                  <span className="text-emerald-700 font-semibold">Exact Match Candidate</span>
                ) : (
                  "Visual Nearest Neighbor"
                )}
              </span>
            </div>

            <div>
              <span className="text-charcoal-500 block mb-0.5">Filename</span>
              <span className="font-mono text-[11px] text-charcoal-700 truncate block">
                {result.filename}
              </span>
            </div>
          </div>

          {/* Action Row */}
          <div className="flex items-center justify-between pt-2">
            <span className="text-[11px] text-charcoal-500">
              Verified against Qdrant index with MobileNetV4 FP32 ONNX
            </span>

            <div className="flex items-center space-x-2">
              {onSearchSimilar && (
                <button
                  type="button"
                  onClick={() => {
                    onSearchSimilar(result.image_url);
                    onClose();
                  }}
                  className="px-4 py-2 text-xs font-medium text-gold-deep bg-gold/15 hover:bg-gold/25 rounded-lg transition flex items-center space-x-1.5"
                >
                  <Sparkles className="w-3.5 h-3.5" />
                  <span>Search with this item</span>
                </button>
              )}
              <button
                type="button"
                onClick={onClose}
                className="px-4 py-2 text-xs font-medium bg-charcoal-900 text-ivory rounded-lg hover:bg-charcoal-800 transition shadow-sm"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
