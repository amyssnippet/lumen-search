"use client";

import React, { useRef, useState } from "react";
import { UploadCloud, Image as ImageIcon, X, Sparkles, CheckCircle2 } from "lucide-react";

interface ImageDropzoneProps {
  selectedFile: File | null;
  previewUrl: string | null;
  onFileSelect: (file: File, suggestedCategory?: string) => void;
  onClear: () => void;
  disabled?: boolean;
}

const PRESET_QUERIES = [
  {
    name: "Floral Cluster (Worn on Hand)",
    url: "/media/ring/ring-downloaded.png",
    category: "ring",
    tag: "Real Worn Photograph",
  },
  {
    name: "Ring Variant 032 (Real Query)",
    url: "/media/query_ring/q_ring_var_001_ring_032.jpg",
    category: "ring",
    tag: "Hard Negative Test",
  },
  {
    name: "Solitaire Diamond Ring",
    url: "/media/ring/ring_001.jpg",
    category: "ring",
    tag: "Catalogue Sample",
  },
  {
    name: "Pendant Necklace",
    url: "/media/necklace/necklace_005.jpg",
    category: "necklace",
    tag: "Catalogue Sample",
  },
  {
    name: "Emerald Cut Ring",
    url: "/media/ring/ring_020.jpg",
    category: "ring",
    tag: "Catalogue Sample",
  },
];

export function ImageDropzone({
  selectedFile,
  previewUrl,
  onFileSelect,
  onClear,
  disabled,
}: ImageDropzoneProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [loadingPreset, setLoadingPreset] = useState<string | null>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    if (!disabled) setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (disabled) return;

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (file.type.startsWith("image/")) {
        onFileSelect(file);
      }
    }
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      onFileSelect(e.target.files[0]);
    }
  };

  const handleSelectPreset = async (preset: (typeof PRESET_QUERIES)[0]) => {
    try {
      setLoadingPreset(preset.name);
      const res = await fetch(preset.url);
      if (!res.ok) throw new Error("Failed to fetch preset sample");
      const blob = await res.blob();
      const filename = preset.url.split("/").pop() || "sample.jpg";
      const file = new File([blob], filename, { type: blob.type || "image/jpeg" });
      onFileSelect(file, preset.category);
    } catch (err) {
      console.error("Could not load preset:", err);
    } finally {
      setLoadingPreset(null);
    }
  };

  return (
    <div className="space-y-4">
      {/* Upload Box / Preview */}
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => !selectedFile && !disabled && fileInputRef.current?.click()}
        className={`relative border-2 border-dashed rounded-2xl transition-all duration-200 overflow-hidden ${
          selectedFile
            ? "border-gold/60 bg-gold-subtle/40 p-4"
            : isDragging
            ? "border-gold bg-gold-subtle/80 cursor-copy py-12 px-6"
            : "border-charcoal-300 hover:border-gold/80 bg-white hover:bg-ivory-50/60 cursor-pointer py-10 px-6"
        } ${disabled ? "opacity-60 cursor-not-allowed" : ""}`}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="image/jpeg,image/png,image/webp,image/jpg"
          className="hidden"
          onChange={handleFileInputChange}
          disabled={disabled}
        />

        {selectedFile && previewUrl ? (
          <div className="flex flex-col sm:flex-row items-center gap-5">
            <div className="relative w-40 h-40 sm:w-48 sm:h-48 rounded-xl overflow-hidden border border-charcoal-200 bg-white shadow-card flex-shrink-0 group">
              <img
                src={previewUrl}
                alt="Selected query"
                className="w-full h-full object-contain"
              />
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  onClear();
                }}
                disabled={disabled}
                className="absolute top-2 right-2 p-1.5 rounded-full bg-charcoal-900/80 text-white hover:bg-charcoal-900 shadow-md transition"
                title="Remove image"
              >
                <X className="w-3.5 h-3.5" />
              </button>
            </div>

            <div className="flex-1 space-y-2 text-center sm:text-left">
              <div className="inline-flex items-center space-x-1.5 px-2.5 py-0.5 rounded-full bg-emerald-50 text-emerald-800 border border-emerald-200 text-xs font-medium">
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
                <span>Image loaded for vector encoding</span>
              </div>
              <h3 className="text-sm font-semibold text-charcoal-900 truncate max-w-sm">
                {selectedFile.name}
              </h3>
              <p className="text-xs text-charcoal-500 font-mono">
                {(selectedFile.size / 1024).toFixed(1)} KB • {selectedFile.type || "image/jpeg"}
              </p>
              <div className="pt-2">
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    fileInputRef.current?.click();
                  }}
                  disabled={disabled}
                  className="text-xs text-gold-deep hover:text-gold-dark font-medium underline"
                >
                  Change image
                </button>
              </div>
            </div>
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center text-center space-y-3">
            <div className="w-14 h-14 rounded-2xl bg-ivory-200 border border-charcoal-200/80 flex items-center justify-center text-gold-deep shadow-sm">
              <UploadCloud className="w-7 h-7" />
            </div>
            <div>
              <p className="text-sm font-medium text-charcoal-800">
                Drag and drop your jewellery photograph here, or{" "}
                <span className="text-gold-deep font-semibold underline decoration-gold/60">
                  browse file
                </span>
              </p>
              <p className="text-xs text-charcoal-500 mt-1">
                Supports real ring & necklace captures (JPEG, PNG, WebP up to 15MB)
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Quick Test Presets */}
      <div>
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs font-semibold text-charcoal-700 uppercase tracking-wider flex items-center">
            <Sparkles className="w-3.5 h-3.5 text-gold-deep mr-1.5" />
            Quick Test Samples (Instant 1-Click Search)
          </span>
          <span className="text-[11px] text-charcoal-500">Real indexed dataset items</span>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
          {PRESET_QUERIES.map((preset) => (
            <button
              key={preset.url}
              type="button"
              onClick={() => handleSelectPreset(preset)}
              disabled={disabled || loadingPreset !== null}
              className="flex items-center space-x-2.5 p-2 rounded-xl bg-white hover:bg-ivory-100 border border-charcoal-200 text-left transition shadow-subtle group hover:border-gold/60 disabled:opacity-50"
            >
              <div className="w-10 h-10 rounded-lg overflow-hidden bg-ivory-200 flex-shrink-0 border border-charcoal-100">
                <img
                  src={preset.url}
                  alt={preset.name}
                  className="w-full h-full object-cover group-hover:scale-105 transition duration-200"
                />
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-xs font-medium text-charcoal-800 truncate group-hover:text-gold-deep">
                  {preset.name}
                </p>
                <p className="text-[10px] text-charcoal-500 font-mono truncate">
                  {preset.tag}
                </p>
              </div>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
