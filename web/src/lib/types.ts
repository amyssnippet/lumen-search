export interface SearchTiming {
  preprocessing: number;
  encoding: number;
  search: number;
  total: number;
}

export interface SearchQuality {
  is_blurry: boolean;
  sharpness_score: number;
  mean_luminance: number;
  is_underexposed: boolean;
  is_overexposed: boolean;
  is_low_res: boolean;
  skin_detected?: boolean;
  skin_ratio?: number;
  warnings: string[];
  applied_steps: string[];
  robust_crops?: string[];
}

export interface SearchRouting {
  target: string;
  fallback: boolean;
  fallback_reason?: string;
}

export interface SearchLocalization {
  box: [number, number, number, number];
  box_pixels: [number, number, number, number];
  label: string;
  confidence: number;
  method: string;
  is_fallback: boolean;
}

export interface SearchResult {
  sku: string;
  category: string;
  similarity: number;
  distance: number;
  filename: string;
  image_url: string;
  metal_color?: string;
  gemstone_type?: string;
  setting_type?: string;
  provenance?: string;
  exact_match_candidate?: boolean;
  match_type?: string;
  confidence_label?: string;
  best_crop?: string;
  global_score?: number;
  chamfer_score?: number;
}

export interface SearchResponse {
  model_name: string;
  pipeline_mode?: string;
  backend: string;
  search_mode: string;
  category_filter?: string;
  robust_mode?: boolean;
  timing_ms: SearchTiming;
  quality: SearchQuality;
  has_reliable_match: boolean;
  top_similarity: number;
  total_candidates: number;
  decision?: "EXACT_MATCH" | "VISUAL_SIMILARITY" | "NO_CONFIDENT_MATCH" | string;
  decision_explanation?: string;
  category_filter_applied?: string;
  detected_category?: string;
  results: SearchResult[];
  routing?: SearchRouting;
  query_id: string;
  localization?: SearchLocalization;
  client_e2e_ms?: number;
}

export interface ProductAsset {
  id: string;
  filename: string;
  image_url: string;
  view_role: string;
  perspective: string;
  provenance_tier?: string;
}

export interface Product {
  id: string;
  sku: string;
  name: string;
  category: string;
  status: string;
  metal_color?: string;
  gemstone_type?: string;
  setting_type?: string;
  price_cents?: number;
  currency?: string;
  asset_count: number;
  thumbnail_url?: string;
  assets?: ProductAsset[];
}

export interface UserIdentity {
  tenant_id: string;
  actor_id: string;
  actor_type: string;
  email?: string;
  role?: string;
  scopes: string[];
  tier?: string;
}

export interface SystemStatus {
  status: string;
  service: string;
  uptime_sec: number;
  dependencies: {
    database: boolean;
    redis: boolean;
    qdrant: boolean;
    onnx_model: boolean;
  };
  lifecycle?: {
    active_production_alias: string;
    target_collection: string;
    canary_enabled: boolean;
    candidate_collection?: string;
    canary_percentage?: number;
  };
  catalog?: {
    total_images_in_dataset: number;
    dataset_categories: { [key: string]: number };
    indexed_points?: number;
  };
}
