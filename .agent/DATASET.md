# Dataset Provenance, Statistics, and Limitations

## 1. Verified Inventory
- **Root Directory**: `Jewellery_Data/`
- **Total Image Count**: 492
- **Corrupted Images**: 0 (all 492 verified readable by Pillow)
- **Category Breakdown**:
  - `necklace`: 301 images (61.2%)
  - `ring`: 191 images (38.8%)
- **Image Formats**:
  - JPEG: 489
  - PNG: 3 (`ring-downloaded.png`, `ring-jay-test.png`, `ring_064.png`)
- **Color Modes**:
  - `RGB`: 490
  - `RGBA`: 1 (`ring-jay-test.png` contains alpha channel)
  - `P` (Paletted): 1 (`ring-downloaded.png`)
- **Resolutions**:
  - Predominant: 262x262 (429 images, 87.2%), 225x225 (27 images, 5.5%)
  - Width range: [90, 275]
  - Height range: [90, 267]
  - Aspect ratios: Mostly 1.0 (square catalog crops), with occasional rectangular thumbnails.

## 2. Duplicate Analysis
### Exact Duplicates (Identical SHA-256 byte hashes):
1. **Set 1**: `Jewellery_Data/necklace/necklace_223.jpg`, `necklace_224.jpg`, `necklace_225.jpg`
2. **Set 2**: `Jewellery_Data/necklace/necklace_233.jpg`, `necklace_234.jpg`
3. **Set 3**: `Jewellery_Data/necklace/necklace_273.jpg`, `necklace_279.jpg`

### Near-Duplicates:
- Perceptual difference hashing (64-bit dHash with Hamming distance threshold <= 2) identified 84 near-duplicate pairs.
- Represents multiple angle shots, slight lighting variations, or re-compressed catalog uploads of identical jewellery pieces.

## 3. Metadata & SKU Identity Status
- **External Metadata**: None. No CSV, JSON, or SQL database accompanied the image folders.
- **Filename Conventions**:
  - Sequential numbered strings: `ring_001.jpg` to `ring_193.jpg`, `necklace_1.jpg` to `necklace_121.jpg`.
  - Missing sequence numbers exist (e.g. `ring_010.jpg` to `ring_012.jpg` are absent).
  - Outlier filenames: `ring-downloaded.png`, `ring-jay-test.png`.
- **Identity Label Coverage**: In the absence of manufacturer SKU numbers, exact duplicates and verified near-duplicate pairs serve as ground-truth identity pairs for exact-match retrieval benchmarking. Singletons are evaluated for category and design-level visual similarity.

## 4. Partitions & Data Leakage Prevention
- **Catalog vs Query Split**:
  - Catalog Index: 432 canonical items.
  - Query Test Set (Historical v1/v2): 60 items (including duplicate/near-duplicate pairs).
  - Exact duplicates and paired multi-view images are explicitly linked so that a query only counts as a hit when retrieving its true twin/counterpart, and identical images do not artificially inflate similarity when self-matching is excluded.

## 5. Ground-Truth Annotations & Balanced Benchmark v3 (EXP-005)
- **Annotation Dataset (`data/annotations.jsonl`)**:
  - 90 initial verified records: 60 confirmed same-SKU, 30 confirmed different-SKU hard negatives.
  - Interactive annotation workflow provided via `/annotate` with synchronized zoom/pan micro-inspection.
- **Balanced Benchmark v3 (`data/evaluation/benchmark_v3_balanced.json`)**:
  - **30 Verified Same-SKU Necklaces** (from raw multi-item clusters).
  - **30 Verified Same-SKU Rings** (4 raw multi-shot identical pairs, 26 verified photographic variations: perspective tilts, studio lighting shifts, sensor noise).
  - **30 Confirmed Hard Negatives** (same-category lookalikes with different SKU identities).
  - **30 Singletons** (zero matching SKU in catalog).
  - All derived query images stored in `data/evaluation/queries/` with documented transformation provenance and strict catalog separation.

## 6. Raw Ring Identity Audit & 4-Gate Ingestion Standard (EXP-008)
- **Ring Identity Scarcity in Raw Catalog**:
  - Comprehensive audit (dHash, SIFT homography, ORB matches, pixel MAE, and DINOv2/v3 feature similarity) across all 191 ring images in `Jewellery_Data/ring/` revealed only **5 natural identity clusters** (`ring_081 <-> ring_108`, `ring_155 <-> ring_172`, `ring_080 <-> ring_164`, `ring_084 <-> ring_181`, `ring_078 <-> ring_079`).
  - High-similarity lookalikes ($> 0.92$ cosine similarity, e.g., `ring_066 <-> ring_100`, `ring_058 <-> ring_060`, `ring_116 <-> ring_187`) are **confirmed different SKU hard negatives** differing in stone count, prongs, or band thickness.
  - Expanding the real-image ring benchmark requires ingesting genuine external physical multi-view captures.
- **Formal 4-Gate Production SKU Verification Standard**:
  1. **Gate 1: ERP/PIM Inventory Lineage**: Validated barcode, SKU ID, and vendor serial number from ERP/PIM inventory database.
  2. **Gate 2: CAD Model & Geometric Specification**: Verified 3D CAD/casting specifications (shank profile, stone carat, prong count) confirming physical identity.
  3. **Gate 3: Turntable Multi-Angle Protocol**: Standardized studio turntable capture ($0^\circ, 45^\circ, 90^\circ$ azimuths, top-down crown view, 45° profile shank view) with recorded camera EXIF, lighting Kelvin, and exposure.
  4. **Gate 4: Non-Transitive Pairwise Validation**: Rejection of graph transitive closure chaining ($A \leftrightarrow B$ and $B \leftrightarrow C \implies A \leftrightarrow C$ is rejected unless explicitly verified).

## 7. Auditable Ingestion Manifest & Reconciled Provenance Accounting (PH-002 & PH-005)
Stored in [`data/ingestion_manifest.json`](file:///Users/amol/Developer/search/data/ingestion_manifest.json) (v2.1.0, schema `ph005.ingestion.v3`):
- **Total Tracked Images**: 518 (492 raw catalog images + 26 synthetic test variation images).
- **Physical Category Breakdown**:
  - `necklace`: 301 raw images
  - `ring`: 191 raw physical images (+ 26 synthetic transforms = 217 ring records total)
- **Accounting Reconciliation Invariant**:
  $$\text{Physical Singletons (423)} + \text{Verified Identical (65)} + \text{Quarantined (4)} = 492 \text{ Raw Catalog Images}$$
  - **Singletons (423)**: 242 necklaces + 181 rings.
  - **Verified Identical (65)**: 55 necklace captures + 10 ring captures (58 independent multi-shot captures + 7 exact byte clone instances).
  - **Quarantined (4)**: `necklace_10`, `necklace_32`, `necklace_91`, `ring_083`.
- **Physical Ring Multi-View Availability**:
  - Total physical ring captures: 191.
  - Physical singletons: 181.
  - Verified physical multi-view SKUs: **5 SKUs** (10 captures: `ring_078`/`079`, `ring_080`/`164`, `ring_081`/`108`, `ring_084`/`181`, `ring_155`/`172`).
  - Target: $\ge 50$ physical ring SKUs.
  - Acquisition Status: **`PARTIALLY_COMPLETE_PENDING_STUDIO_ASSETS`** (5 verified, 45 pending physical studio photoshoot).
- **4-Gate Provenance Status**:
  - Gate 1 (ERP/PIM Lineage): 0.0% coverage (authoritative ERP metadata absent in legacy catalog; zero fake SKUs invented; marked `UNASSIGNED_LEGACY_ERP`).
  - Gate 2 (CAD Model & Geometry): 0.0% coverage.
  - Gate 3 (360° Turntable Protocol): 0.0% coverage on legacy items.
  - Gate 4 (Non-Transitive Pairwise Verification): 99.23% compliance (verified pairwise links only; transitive errors like `ring_083` quarantined).
- **Automated Manifest Validator**:
  - Verified via [`scripts/update_and_validate_ph005_manifest.py`](file:///Users/amol/Developer/search/scripts/update_and_validate_ph005_manifest.py) and [`output/ph005_manifest_integrity_audit.json`](file:///Users/amol/Developer/search/output/ph005_manifest_integrity_audit.json) (0 errors found).

## 8. Physical Studio Acquisition Standard (PH-005 SOP)
Detailed in [`PH-005-studio-acquisition-SOP.md`](file:///Users/amol/Developer/search/PH-005-studio-acquisition-SOP.md):
- **Canonical View Angles**: `crown` ($0^\circ$ normal), `angle_45` ($45^\circ$ oblique 3/4), `profile` ($90^\circ$ side elevation), `shank` (shank hallmark / under-gallery).
- **Optics & Lighting**: 1:1 macro reproduction ratio, cross-polarized continuous LED dome ($5000\text{ K} \pm 100\text{ K}$, $\text{CRI} \ge 98$).
- **Exposure Invariants**: Highlight clipping $< 0.5\%$ blown pixels; black clip floor $> 5$.



