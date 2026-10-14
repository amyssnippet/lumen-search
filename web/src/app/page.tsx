"use client";

import React, { useState } from "react";
import { ImageDropzone } from "@/components/ImageDropzone";
import { FilterBar, SearchOptions } from "@/components/FilterBar";
import { LatencyBreakdown } from "@/components/LatencyBreakdown";
import { SearchResults } from "@/components/SearchResults";
import { SearchResponse } from "@/lib/types";
import { Sparkles, Search, RotateCcw, AlertCircle } from "lucide-react";

export default function SearchPage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [searchOptions, setSearchOptions] = useState<SearchOptions>({
    category: "all",
    searchMode: "similar",
    topK: 10,
    robustMode: true,
    enableClahe: false,
    enableDenoise: false,
    autoEnhance: false,
  });
  const [searchResponse, setSearchResponse] = useState<SearchResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFileSelect = (file: File, suggestedCategory?: string) => {
    setSelectedFile(file);
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    setError(null);
    const effectiveOptions = suggestedCategory
      ? { ...searchOptions, category: suggestedCategory }
      : searchOptions;
    if (suggestedCategory) {
      setSearchOptions(effectiveOptions);
    }
    // Automatically trigger search for smooth user flow
    executeSearch(file, effectiveOptions);
  };

  const handleClear = () => {
    setSelectedFile(null);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(null);
    setSearchResponse(null);
    setError(null);
  };

  const executeSearch = async (file: File, options: SearchOptions) => {
    setIsLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("category_filter", options.category);
      formData.append("search_mode", options.searchMode);
      formData.append("top_k", options.topK.toString());
      formData.append("robust_mode", options.robustMode.toString());
      formData.append("enable_clahe", options.enableClahe.toString());
      formData.append("enable_denoise", options.enableDenoise.toString());
      formData.append("auto_enhance", options.autoEnhance.toString());

      const res = await fetch("/api/search", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.error || "Search execution failed");
      }

      setSearchResponse(data);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSearchWithCandidate = async (imageUrl: string) => {
    try {
      setIsLoading(true);
      setError(null);
      const res = await fetch(imageUrl);
      if (!res.ok) throw new Error("Could not load candidate image");
      const blob = await res.blob();
      const fname = imageUrl.split("/").pop() || "query.jpg";
      const file = new File([blob], fname, { type: blob.type || "image/jpeg" });
      const suggestedCategory = imageUrl.includes("/necklace/") ? "necklace" : imageUrl.includes("/ring/") ? "ring" : undefined;

      setSelectedFile(file);
      const url = URL.createObjectURL(blob);
      setPreviewUrl(url);
      const effectiveOptions = suggestedCategory ? { ...searchOptions, category: suggestedCategory } : searchOptions;
      if (suggestedCategory) {
        setSearchOptions(effectiveOptions);
      }
      await executeSearch(file, effectiveOptions);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleManualSearch = () => {
    if (selectedFile) {
      executeSearch(selectedFile, searchOptions);
    }
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-8 space-y-8">
      {/* Title & Introduction */}
      <div className="text-center space-y-2 max-w-2xl mx-auto">
        <div className="inline-flex items-center space-x-1.5 px-3 py-1 rounded-full bg-gold/15 text-gold-deep text-xs font-semibold tracking-wide uppercase">
          <Sparkles className="w-3.5 h-3.5" />
          <span>Interactive Visual Search</span>
        </div>
        <h1 className="text-3xl sm:text-4xl font-serif font-bold text-charcoal-900 tracking-tight">
          Visual Jewellery Discovery
        </h1>
        <p className="text-sm text-charcoal-600">
          Upload any ring or necklace photograph to instantly match against our high-precision indexed catalogue using MobileNetV4 1280-dim representations.
        </p>
      </div>

      {/* Main Search Panel */}
      <div className="space-y-4">
        {/* Upload Zone */}
        <ImageDropzone
          selectedFile={selectedFile}
          previewUrl={previewUrl}
          onFileSelect={handleFileSelect}
          onClear={handleClear}
          disabled={isLoading}
        />

        {/* Filter & Control Bar */}
        <FilterBar
          options={searchOptions}
          onChange={(newOpts) => {
            setSearchOptions(newOpts);
            if (selectedFile) {
              executeSearch(selectedFile, newOpts);
            }
          }}
          disabled={isLoading}
        />

        {/* Action button if file is present */}
        {selectedFile && (
          <div className="flex items-center justify-end space-x-3 pt-1">
            <button
              type="button"
              onClick={handleClear}
              disabled={isLoading}
              className="px-4 py-2 text-xs font-medium text-charcoal-600 hover:text-charcoal-900 hover:bg-charcoal-100 rounded-xl transition flex items-center space-x-1.5"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              <span>Reset</span>
            </button>
            <button
              type="button"
              onClick={handleManualSearch}
              disabled={isLoading}
              className="px-6 py-2.5 text-xs font-medium bg-charcoal-900 text-ivory rounded-xl hover:bg-charcoal-800 transition disabled:opacity-50 shadow-sm flex items-center space-x-2"
            >
              <Search className="w-4 h-4 text-gold" />
              <span>{isLoading ? "Searching..." : "Re-run Search"}</span>
            </button>
          </div>
        )}
      </div>

      {/* Error Alert */}
      {error && (
        <div className="p-4 bg-red-50 border border-red-200 rounded-2xl flex items-center space-x-3 text-sm text-red-800">
          <AlertCircle className="w-5 h-5 flex-shrink-0 text-red-600" />
          <div className="flex-1">
            <p className="font-semibold">Search Request Error</p>
            <p className="text-xs text-red-700 mt-0.5">{error}</p>
          </div>
        </div>
      )}

      {/* Latency & Quality Breakdown */}
      {searchResponse && (
        <LatencyBreakdown
          timing={searchResponse.timing_ms}
          clientE2E={searchResponse.client_e2e_ms}
          quality={searchResponse.quality}
          hasReliableMatch={searchResponse.has_reliable_match}
          topSimilarity={searchResponse.top_similarity}
          decision={searchResponse.decision}
          decisionExplanation={searchResponse.decision_explanation}
          categoryFilterApplied={searchResponse.category_filter_applied}
          detectedCategory={searchResponse.detected_category}
        />
      )}

      {/* Retrieved Candidates Grid */}
      <SearchResults
        response={searchResponse}
        queryPreviewUrl={previewUrl}
        onSearchWithItem={handleSearchWithCandidate}
        isLoading={isLoading}
      />
    </div>
  );
}
