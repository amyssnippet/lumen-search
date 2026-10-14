"use client";

import React, { useEffect, useState } from "react";
import { Product } from "@/lib/types";
import { Database, Search, Sparkles, ChevronLeft, ChevronRight, Eye, Diamond } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";

export default function CataloguePage() {
  const router = useRouter();
  const [products, setProducts] = useState<Product[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [category, setCategory] = useState("all");
  const [skuQuery, setSkuQuery] = useState("");
  const [page, setPage] = useState(1);
  const limit = 24;

  const fetchProducts = async (cat: string, sku: string, p: number) => {
    setLoading(true);
    setError(null);
    try {
      const offset = (p - 1) * limit;
      const params = new URLSearchParams({
        category: cat,
        limit: limit.toString(),
        offset: offset.toString(),
      });
      if (sku.trim()) {
        params.set("sku", sku.trim());
      }

      const res = await fetch(`/api/products?${params.toString()}`);
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.error || "Failed to fetch catalogue products");
      }

      setProducts(data.products || []);
      setTotal(data.total || 0);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchProducts(category, skuQuery, page);
  }, [category, page]);

  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setPage(1);
    fetchProducts(category, skuQuery, 1);
  };

  const totalPages = Math.ceil(total / limit) || 1;

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-8 space-y-8">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-4 border-b border-charcoal-200 pb-6">
        <div>
          <div className="inline-flex items-center space-x-1.5 px-3 py-1 rounded-full bg-emerald-50 text-emerald-800 text-xs font-semibold tracking-wide uppercase mb-2">
            <Database className="w-3.5 h-3.5" />
            <span>Indexed Catalogue</span>
          </div>
          <h1 className="text-3xl font-serif font-bold text-charcoal-900 tracking-tight">
            Jewellery Catalogue Browser
          </h1>
          <p className="text-xs text-charcoal-600 mt-1">
            Browse and search relational inventory and image assets indexed in Qdrant vector database.
          </p>
        </div>

        {/* Counter */}
        <div className="text-right">
          <span className="text-2xl font-bold font-mono text-charcoal-900">{total}</span>
          <span className="text-xs text-charcoal-500 block">Total Items Found</span>
        </div>
      </div>

      {/* Filter and Search Bar */}
      <div className="bg-white rounded-2xl border border-charcoal-200 p-4 shadow-subtle flex flex-col sm:flex-row items-center justify-between gap-4">
        {/* Category Selector */}
        <div className="flex items-center space-x-1.5 bg-ivory-100 p-1 rounded-xl border border-charcoal-200/80 w-full sm:w-auto">
          {[
            { id: "all", label: "All Items" },
            { id: "ring", label: "Rings" },
            { id: "necklace", label: "Necklaces" },
          ].map((c) => (
            <button
              key={c.id}
              onClick={() => {
                setCategory(c.id);
                setPage(1);
              }}
              className={`flex-1 sm:flex-none px-3.5 py-1.5 text-xs font-medium rounded-lg transition ${
                category === c.id
                  ? "bg-charcoal-900 text-ivory shadow-sm"
                  : "text-charcoal-600 hover:text-charcoal-900 hover:bg-ivory-200"
              }`}
            >
              {c.label}
            </button>
          ))}
        </div>

        {/* SKU search */}
        <form onSubmit={handleSearchSubmit} className="flex items-center space-x-2 w-full sm:w-80">
          <div className="relative flex-1">
            <Search className="w-4 h-4 text-charcoal-400 absolute left-3 top-2.5" />
            <input
              type="text"
              value={skuQuery}
              onChange={(e) => setSkuQuery(e.target.value)}
              placeholder="Search by SKU (e.g. RING-032)..."
              className="w-full pl-9 pr-3 py-1.5 text-xs border border-charcoal-200 rounded-xl bg-ivory-50 text-charcoal-900 focus:outline-none focus:ring-1 focus:ring-gold"
            />
          </div>
          <button
            type="submit"
            className="px-3.5 py-1.5 text-xs font-medium bg-charcoal-900 text-ivory rounded-xl hover:bg-charcoal-800 transition"
          >
            Filter
          </button>
        </form>
      </div>

      {/* Error state */}
      {error && (
        <div className="p-4 bg-red-50 border border-red-200 rounded-xl text-xs text-red-700">
          {error}
        </div>
      )}

      {/* Product Grid */}
      {loading ? (
        <div className="bg-white rounded-2xl border border-charcoal-200 p-16 text-center space-y-3">
          <div className="w-10 h-10 rounded-full border-4 border-gold border-t-transparent animate-spin mx-auto" />
          <p className="text-xs text-charcoal-500 font-mono">Loading catalogue records...</p>
        </div>
      ) : products.length === 0 ? (
        <div className="bg-white rounded-2xl border border-charcoal-200 p-16 text-center space-y-2">
          <p className="text-sm font-semibold text-charcoal-800">No catalogue items match your criteria</p>
          <p className="text-xs text-charcoal-500">Try clearing the SKU filter or selecting another category.</p>
        </div>
      ) : (
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-4">
          {products.map((p) => {
            const thumb = p.thumbnail_url || (p.assets && p.assets[0]?.image_url);
            return (
              <div
                key={p.id}
                className="group bg-white rounded-2xl border border-charcoal-200 overflow-hidden hover:shadow-card hover:border-gold/60 transition flex flex-col justify-between"
              >
                {/* Thumbnail */}
                <div className="relative aspect-square w-full bg-ivory-100 overflow-hidden">
                  {thumb ? (
                    <img
                      src={thumb}
                      alt={p.sku}
                      className="w-full h-full object-cover group-hover:scale-105 transition duration-300"
                      loading="lazy"
                    />
                  ) : (
                    <div className="w-full h-full flex items-center justify-center text-charcoal-400">
                      No Image
                    </div>
                  )}
                  <span className="absolute top-2 left-2 text-[10px] font-mono bg-charcoal-900/80 text-ivory px-2 py-0.5 rounded backdrop-blur-sm">
                    {p.sku}
                  </span>
                </div>

                {/* Info */}
                <div className="p-3 space-y-2">
                  <div>
                    <h3 className="text-xs font-semibold text-charcoal-900 truncate">
                      {p.name || p.sku}
                    </h3>
                    <p className="text-[10px] text-charcoal-500 uppercase font-mono mt-0.5">
                      {p.category} {p.metal_color ? `• ${p.metal_color}` : ""}
                    </p>
                  </div>

                  {/* Actions */}
                  <div className="pt-2 border-t border-charcoal-100 flex items-center justify-between text-xs">
                    {thumb && (
                      <Link
                        href={`/?preset=${encodeURIComponent(thumb)}`}
                        className="text-[11px] text-gold-deep hover:text-gold-dark font-medium flex items-center space-x-1"
                        title="Search with this item"
                      >
                        <Sparkles className="w-3 h-3" />
                        <span>Search Similar</span>
                      </Link>
                    )}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Pagination Controls */}
      <div className="flex items-center justify-between pt-4 border-t border-charcoal-200 text-xs">
        <span className="text-charcoal-500">
          Page <strong>{page}</strong> of <strong>{totalPages}</strong>
        </span>

        <div className="flex items-center space-x-2">
          <button
            type="button"
            disabled={page <= 1 || loading}
            onClick={() => setPage(page - 1)}
            className="px-3 py-1.5 border border-charcoal-200 rounded-lg bg-white text-charcoal-700 hover:bg-ivory-100 disabled:opacity-40 transition flex items-center space-x-1"
          >
            <ChevronLeft className="w-3.5 h-3.5" />
            <span>Previous</span>
          </button>
          <button
            type="button"
            disabled={page >= totalPages || loading}
            onClick={() => setPage(page + 1)}
            className="px-3 py-1.5 border border-charcoal-200 rounded-lg bg-white text-charcoal-700 hover:bg-ivory-100 disabled:opacity-40 transition flex items-center space-x-1"
          >
            <span>Next</span>
            <ChevronRight className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>
    </div>
  );
}
