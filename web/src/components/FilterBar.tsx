"use client";

import React, { useState } from "react";
import { SlidersHorizontal, ChevronDown, ChevronUp, Cpu, Database } from "lucide-react";

export interface SearchOptions {
  category: string;
  searchMode: "similar" | "exact";
  pipelineMode: "baseline" | "v2_dinov3";
  topK: number;
  robustMode: boolean;
  enableClahe: boolean;
  enableDenoise: boolean;
  autoEnhance: boolean;
  metalColor?: string;
  gemstoneType?: string;
}

interface FilterBarProps {
  options: SearchOptions;
  onChange: (options: SearchOptions) => void;
  disabled?: boolean;
}

export function FilterBar({ options, onChange, disabled }: FilterBarProps) {
  const [showAdvanced, setShowAdvanced] = useState(false);

  const categories = [
    { id: "all", label: "All Categories" },
    { id: "ring", label: "Rings" },
    { id: "necklace", label: "Necklaces" },
  ];

  return (
    <div className="bg-white rounded-2xl border border-charcoal-200 p-4 shadow-subtle space-y-4">
      {/* Primary Quick Controls */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        {/* Pipeline Architecture Selector (Baseline vs Fine-Grained v2) */}
        <div className="flex items-center space-x-1 bg-ivory-100 p-1 rounded-xl border border-charcoal-200/80 text-xs">
          <button
            type="button"
            disabled={disabled}
            onClick={() => onChange({ ...options, pipelineMode: "baseline" })}
            className={`px-3 py-1.5 font-medium rounded-lg transition ${
              options.pipelineMode === "baseline"
                ? "bg-charcoal-900 text-ivory shadow-sm"
                : "text-charcoal-600 hover:text-charcoal-900 hover:bg-ivory-200"
            }`}
          >
            Baseline (MobileNetV4)
          </button>
          <button
            type="button"
            disabled={disabled}
            onClick={() => onChange({ ...options, pipelineMode: "v2_dinov3" })}
            className={`px-3 py-1.5 font-medium rounded-lg transition flex items-center space-x-1.5 ${
              options.pipelineMode === "v2_dinov3"
                ? "bg-gold text-white shadow-sm"
                : "text-charcoal-600 hover:text-charcoal-900 hover:bg-ivory-200"
            }`}
          >
            <span>Fine-Grained v2 (DINOv3)</span>
            <span className="text-[10px] bg-white/20 px-1 py-0.5 rounded font-mono font-semibold">POC</span>
          </button>
        </div>

        {/* Category Pills */}
        <div className="flex items-center space-x-1.5 bg-ivory-100 p-1 rounded-xl border border-charcoal-200/80">
          {categories.map((c) => {
            const isSelected = options.category === c.id;
            return (
              <button
                key={c.id}
                type="button"
                disabled={disabled}
                onClick={() => onChange({ ...options, category: c.id })}
                className={`px-3 py-1.5 text-xs font-medium rounded-lg transition ${
                  isSelected
                    ? "bg-charcoal-900 text-ivory shadow-sm"
                    : "text-charcoal-600 hover:text-charcoal-900 hover:bg-ivory-200"
                } disabled:opacity-50`}
              >
                {c.label}
              </button>
            );
          })}
        </div>

        {/* Search Mode Toggle */}
        <div className="flex items-center space-x-3">
          <div className="flex items-center space-x-1 bg-ivory-100 p-1 rounded-xl border border-charcoal-200/80 text-xs">
            <button
              type="button"
              disabled={disabled}
              onClick={() => onChange({ ...options, searchMode: "similar" })}
              className={`px-3 py-1.5 font-medium rounded-lg transition ${
                options.searchMode === "similar"
                  ? "bg-gold text-white shadow-sm"
                  : "text-charcoal-600 hover:text-charcoal-900"
              }`}
            >
              Visual Similarity
            </button>
            <button
              type="button"
              disabled={disabled}
              onClick={() => onChange({ ...options, searchMode: "exact" })}
              className={`px-3 py-1.5 font-medium rounded-lg transition ${
                options.searchMode === "exact"
                  ? "bg-charcoal-900 text-ivory shadow-sm"
                  : "text-charcoal-600 hover:text-charcoal-900"
              }`}
            >
              Exact Match
            </button>
          </div>

          {/* Toggle Advanced Button */}
          <button
            type="button"
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="flex items-center space-x-1 px-3 py-1.5 text-xs font-medium text-charcoal-700 bg-ivory-100 hover:bg-charcoal-100 border border-charcoal-200 rounded-xl transition"
          >
            <SlidersHorizontal className="w-3.5 h-3.5 text-gold-deep" />
            <span>Options</span>
            {showAdvanced ? (
              <ChevronUp className="w-3 h-3 text-charcoal-500" />
            ) : (
              <ChevronDown className="w-3 h-3 text-charcoal-500" />
            )}
          </button>
        </div>
      </div>

      {/* Advanced Drawer */}
      {showAdvanced && (
        <div className="pt-3 border-t border-charcoal-100 grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4 animate-in fade-in duration-150 text-xs">
          {/* Top K */}
          <div>
            <label className="block text-charcoal-600 font-medium mb-1">Results Limit (Top K)</label>
            <select
              disabled={disabled}
              value={options.topK}
              onChange={(e) => onChange({ ...options, topK: Number(e.target.value) })}
              className="w-full px-2.5 py-1.5 border border-charcoal-200 rounded-lg bg-ivory-50 text-charcoal-800 focus:outline-none focus:ring-1 focus:ring-gold"
            >
              <option value={5}>Top 5 Candidates</option>
              <option value={10}>Top 10 Candidates</option>
              <option value={20}>Top 20 Candidates</option>
            </select>
          </div>

          {/* Engine Arch indicator */}
          <div>
            <label className="block text-charcoal-600 font-medium mb-1">Retriever & Vector Store</label>
            <div className="px-2.5 py-1.5 bg-ivory-100 border border-charcoal-200 rounded-lg text-charcoal-700 flex items-center justify-between font-mono text-[11px]">
              <span className="flex items-center">
                <Cpu className="w-3 h-3 text-gold-deep mr-1" />
                {options.pipelineMode === "v2_dinov3" ? "DINOv3 ViT-S/16 INT8" : "MobileNetV4 FP32 ONNX"}
              </span>
              <span className="flex items-center">
                <Database className="w-3 h-3 text-emerald-600 mr-1" />
                {options.pipelineMode === "v2_dinov3" ? "Qdrant (v2 isolated)" : "Qdrant (staging prod)"}
              </span>
            </div>
          </div>

          {/* Image Preprocessing Toggles */}
          <div className="sm:col-span-2 flex flex-wrap items-center gap-3 pt-4 sm:pt-6">
            <label className="flex items-center space-x-1.5 cursor-pointer">
              <input
                type="checkbox"
                disabled={disabled}
                checked={options.robustMode}
                onChange={(e) => onChange({ ...options, robustMode: e.target.checked })}
                className="rounded border-charcoal-300 text-gold focus:ring-gold"
              />
              <span className="text-charcoal-900 font-medium">Adaptive Worn-Item & Multi-Crop Retrieval</span>
            </label>

            <label className="flex items-center space-x-1.5 cursor-pointer">
              <input
                type="checkbox"
                disabled={disabled}
                checked={options.enableClahe}
                onChange={(e) => onChange({ ...options, enableClahe: e.target.checked })}
                className="rounded border-charcoal-300 text-gold focus:ring-gold"
              />
              <span className="text-charcoal-700">CLAHE Equalization</span>
            </label>

            <label className="flex items-center space-x-1.5 cursor-pointer">
              <input
                type="checkbox"
                disabled={disabled}
                checked={options.enableDenoise}
                onChange={(e) => onChange({ ...options, enableDenoise: e.target.checked })}
                className="rounded border-charcoal-300 text-gold focus:ring-gold"
              />
              <span className="text-charcoal-700">Bilateral Denoise</span>
            </label>

            <label className="flex items-center space-x-1.5 cursor-pointer">
              <input
                type="checkbox"
                disabled={disabled}
                checked={options.autoEnhance}
                onChange={(e) => onChange({ ...options, autoEnhance: e.target.checked })}
                className="rounded border-charcoal-300 text-gold focus:ring-gold"
              />
              <span className="text-charcoal-700">Auto-Enhance</span>
            </label>
          </div>
        </div>
      )}
    </div>
  );
}
