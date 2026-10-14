"use client";

import React from "react";
import { Sparkles, Maximize2, ShieldCheck, Tag } from "lucide-react";
import { SearchResult } from "@/lib/types";

interface ResultCardProps {
  result: SearchResult;
  rank: number;
  onInspect: (result: SearchResult) => void;
  onSearchWithItem?: (imageUrl: string) => void;
}

export function ResultCard({
  result,
  rank,
  onInspect,
  onSearchWithItem,
}: ResultCardProps) {
  const simPct = (result.similarity * 100).toFixed(1);
  const isExactCandidate = result.exact_match_candidate || result.similarity >= 0.88;
  const isHighMatch = result.similarity >= 0.70;
  const isSimilarDesign = result.similarity >= 0.52;

  const getBadgeStyle = () => {
    if (isExactCandidate) {
      return {
        classes: "bg-emerald-600/95 text-white",
        label: "Exact SKU Candidate",
        showShield: true,
      };
    }
    if (isHighMatch) {
      return {
        classes: "bg-teal-600/95 text-white",
        label: "High Similarity Design",
        showShield: false,
      };
    }
    if (isSimilarDesign) {
      return {
        classes: "bg-amber-600/95 text-white",
        label: "Visual Style Candidate",
        showShield: false,
      };
    }
    return {
      classes: "bg-charcoal-500/90 text-white",
      label: "Low Similarity",
      showShield: false,
    };
  };

  const badge = getBadgeStyle();

  return (
    <div className="group bg-white rounded-2xl border border-charcoal-200 overflow-hidden hover:shadow-card hover:border-gold/60 transition-all duration-200 flex flex-col">
      {/* Image Thumbnail Container */}
      <div className="relative aspect-square w-full bg-ivory-100 overflow-hidden cursor-pointer" onClick={() => onInspect(result)}>
        <img
          src={result.image_url}
          alt={result.sku}
          className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
          loading="lazy"
        />

        {/* Rank Badge */}
        <div className="absolute top-2.5 left-2.5 px-2 py-0.5 rounded-md bg-charcoal-900/80 backdrop-blur-sm text-ivory text-[10px] font-mono font-medium">
          #{rank}
        </div>

        {/* Similarity Score Badge */}
        <div
          title={`${result.confidence_label || badge.label} (Cosine similarity: ${simPct}%)`}
          className={`absolute top-2.5 right-2.5 px-2.5 py-0.5 rounded-full text-xs font-mono font-bold shadow-sm backdrop-blur-sm flex items-center space-x-1 ${badge.classes}`}
        >
          {badge.showShield && <ShieldCheck className="w-3 h-3 mr-0.5" />}
          <span>{simPct}%</span>
        </div>

        {/* Hover Inspect Overlay */}
        <div className="absolute inset-0 bg-black/20 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center pointer-events-none">
          <div className="px-3 py-1.5 rounded-lg bg-charcoal-900/90 text-white text-xs font-medium flex items-center space-x-1.5 shadow-lg">
            <Maximize2 className="w-3.5 h-3.5" />
            <span>Inspect</span>
          </div>
        </div>
      </div>

      {/* Details Box */}
      <div className="p-3.5 flex-1 flex flex-col justify-between space-y-3">
        <div>
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold text-charcoal-900 font-mono tracking-tight truncate">
              {result.sku}
            </span>
            <span className="text-[10px] uppercase font-semibold tracking-wider text-charcoal-500 bg-ivory-200 px-2 py-0.5 rounded">
              {result.category}
            </span>
          </div>

          {/* Confidence Description */}
          {result.confidence_label && (
            <p className="text-[11px] font-medium text-charcoal-600 mt-1 truncate">
              {result.confidence_label}
            </p>
          )}

          {/* Metadata Pills */}
          <div className="flex flex-wrap gap-1 mt-2">
            {result.metal_color && (
              <span className="text-[10px] text-charcoal-600 bg-ivory-100 px-1.5 py-0.5 rounded border border-charcoal-100">
                {result.metal_color}
              </span>
            )}
            {result.gemstone_type && (
              <span className="text-[10px] text-charcoal-600 bg-ivory-100 px-1.5 py-0.5 rounded border border-charcoal-100">
                {result.gemstone_type}
              </span>
            )}
            {result.setting_type && (
              <span className="text-[10px] text-charcoal-600 bg-ivory-100 px-1.5 py-0.5 rounded border border-charcoal-100">
                {result.setting_type}
              </span>
            )}
          </div>
        </div>

        {/* Action Row */}
        <div className="pt-2 border-t border-charcoal-100 flex items-center justify-between text-xs">
          <button
            type="button"
            onClick={() => onInspect(result)}
            className="text-charcoal-600 hover:text-charcoal-900 font-medium"
          >
            Details
          </button>

          {onSearchWithItem && (
            <button
              type="button"
              onClick={() => onSearchWithItem(result.image_url)}
              className="flex items-center space-x-1 text-gold-deep hover:text-gold-dark font-medium group/btn"
              title="Run similarity search using this candidate image"
            >
              <Sparkles className="w-3 h-3" />
              <span>Find Similar</span>
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
