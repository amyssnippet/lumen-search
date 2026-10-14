# Standard Operating Procedure (SOP): Multi-Angle Studio Acquisition for Fine Jewellery Rings

**Document ID**: SOP-PH005-RING-ACQ-v1.0  
**Effective Date**: 2026-10-11  
**Target Milestone**: PH-005 Physical Ring Data Acquisition & Studio Ingestion Standard  
**Target Scope**: Minimum 50 unique physical ring SKUs with $\ge 4$ genuine multi-view captures per SKU ($\ge 200$ high-fidelity photographs)  

---

## 1. Purpose & Guiding Principles

The objective of this Standard Operating Procedure is to govern the physical acquisition and metadata curation of multi-view photographs for fine jewellery rings. This standard guarantees that images ingested into the Jewellery Visual Retrieval Engine (`jewellery_catalog_mobilenetv4_v1`) possess verifiable physical provenance, high optical fidelity, and multi-view geometric coverage.

### 1.1 Invariant Governance Rules
1. **Physical Authenticity**: Every photographic asset must represent a physical, tangible piece of jewellery captured under controlled optical conditions.
2. **Zero Synthetic Substitution**: Synthetic augmentations (rotations, color jitter, affine warps, perspective transforms), digital crops, and repeated exports from identical source captures **SHALL NOT** be counted as independent physical products or independent photographic captures.
3. **Traceable Identity Lineage**: No asset may be ingested without an explicit linkage to an authoritative inventory barcode/SKU, ERP batch record, or CAD engineering manifest. Transitive identity assumptions ("Item A looks like Item B, so Item B is SKU-A") are strictly prohibited.

---

## 2. Multi-Angle Viewpoint Specifications

For each physical ring SKU, a minimum of **four mandatory canonical viewpoints** must be photographed. Optional detail macros may be acquired for complex architectural features.

```
       [Angle 1: Crown View (90° Overhead)]
                      |
                      v
            +-------------------+
            |    ( Gemstone )   |
            +---------+---------+
                     / \
                    /   \
   [Angle 2: 45°]  /     \  [Angle 3: Profile (0° Horizontal)]
                  +-------+
                  | Shank |
                  +-------+
                      |
                      v
      [Angle 4: Shank / Hallmark View]
```

### 2.1 Canonical Angles

| Angle ID | Viewpoint Name | Camera Angle | Focus & Framing Objective | Critical Distinguishing Details Captured |
|---|---|:---:|---|---|
| **VIEW-01** | **Crown / Top View** | 90° normal to table facet | Direct overhead alignment centered on the primary stone table. | Table facet dimensions, halo architecture, prong count/symmetry, side-stone pave rows, crown perimeter outline. |
| **VIEW-02** | **45-Degree Perspective View** | 45° oblique elevation | Three-quarter isometric perspective with ring tilted forward. | Gallery wirework, shoulder stone transition, bezel height, depth of basket relative to shank. |
| **VIEW-03** | **Profile / Side View** | 0° horizontal elevation | True side elevation parallel to finger-hole plane. | Basket trellis, cathedral arches, collet setting, bridge thickness, peg head junction, under-gallery height. |
| **VIEW-04** | **Shank / Underside View** | 180° inverted / oblique | Focus placed on inner shank curvature and under-stone gallery. | Hallmark stamping (karat/metal purity, maker's mark), ajour pierced pattern, inner shank profile (comfort-fit vs flat). |
| **VIEW-05** *(Optional)* | **Prong / Gallery Macro** | Micro-elevation (Macro 2:1) | Extreme close-up of center stone setting. | Claw vs round prong finishing, v-tip corner prongs on princess/cushion cuts, hidden halo micro-pave. |

---

## 3. Optical & Environmental Capture Standards

### 3.1 Hardware & Optical Rig
- **Camera Sensor**: Full-frame or APS-C digital camera ($\ge 24.0$ Megapixels).
- **Lens**: True 1:1 macro lens (focal length $90\text{mm} - 105\text{mm}$) to eliminate wide-angle barrel distortion.
- **Aperture & Focus**:
  - Base aperture: $f/11$ to $f/16$ for optimal optical sharpness without excessive diffraction softening.
  - **Focus-Stacking Mandatory for High-Profile Rings**: For rings with crown depth exceeding $6\text{mm}$, capture an automated 5-to-12 slice focus bracket and combine via exposure-preserving pyramid stacking.
- **Color Space & Bit Depth**: 14-bit RAW capture, exported to 8-bit sRGB JPEG/PNG.
- **Resolution**: Minimum $1024 \times 1024$ pixels; target $2048 \times 2048$ pixels square crop centered on the item.

### 3.2 Lighting & Specular Reflection Control
- **Light Source**: High-CRI continuous LED ($\text{CRI} \ge 96$, $\text{TLCI} \ge 98$, color temperature $5000\text{K} \pm 200\text{K}$).
- **Diffuser Geometry**: Complete $360^\circ$ cylindrical or hemispherical diffusion dome (light cone) enclosing the ring to avoid hard black shadows.
- **Cross-Polarization**: Linear polarizer on light sources and circular polarizer on the macro lens to suppress harsh white specular glare on diamond facet tables while preserving natural gemstone body color.
- **Exposure Calibration**: Exposure calibrated against an 18% neutral gray card. Mean luminance must reside strictly within $180 \le \mu \le 225$ (8-bit scale). Highlight clipping on stone facets must not exceed 2.0% of total mask area.

### 3.3 Ring Staging & Mounting
- **Support Medium**: Transparent non-reflective acrylic prong stand or ultra-low residue staging putty concealed behind the lower shank.
- **Dust & Fingerprint Protocol**: All rings must be cleaned in an ultrasonic bath or micro-fiber alcohol wipe immediately prior to staging. Staging operators must wear lint-free nylon gloves.

---

## 4. Metadata Schema & Capture Attributes

Every captured asset must be serialized into a machine-readable record adhering to the following schema:

```json
{
  "$schema": "https://jewellery-search.org/schemas/ph005-ring-capture.v1.json",
  "asset_id": "RING_SKU001_VIEW01_CROWN",
  "sku_id": "SKU-RING-001",
  "identity_group_id": "PHYS-RING-001",
  "category": "ring",
  "view_angle": "crown_90",
  "capture_session_id": "STUDIO_SESSION_20261011_01",
  "camera_id": "SONY_A7R4_MACRO90_RIG1",
  "lighting_preset": "DIFFUSE_DOME_5000K_POLARIZED",
  "is_focus_stacked": true,
  "focus_slices_count": 8,
  "optical_metrics": {
    "sharpness_laplacian_var": 342.5,
    "mean_luminance": 210.4,
    "highlight_clipped_pct": 0.42
  },
  "physical_attributes": {
    "metal_color": "white_gold",
    "metal_purity": "18k",
    "gemstone_type": "diamond",
    "gemstone_shape": "round_brilliant",
    "setting_type": "solitaire_prong",
    "shank_style": "tapered_comfort_fit",
    "hallmark_inscription": "750_D075"
  },
  "four_gate_provenance": {
    "gate1_erp_verified": true,
    "gate2_cad_matched": true,
    "gate3_turntable_conforming": true,
    "gate4_pairwise_audited": true,
    "erp_po_reference": "PO-2026-JEW-88392",
    "cad_model_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
  },
  "file_info": {
    "rel_path": "Jewellery_Data/ring/SKU-RING-001_crown.jpg",
    "sha256": "8a515c68f62eed3ff1944e4f12f7028b44aa0978082f44c92a9c88d68dff2c49",
    "width": 1024,
    "height": 1024,
    "file_size_bytes": 284910
  }
}
```

---

## 5. Quality Assurance Checklist

Before an acquired ring set is accepted into the ingestion queue, the QA inspector must complete the following inspection gates:

- [ ] **Angle Completeness**: All 4 mandatory angles (`crown`, `angle_45`, `profile`, `shank`) are present for the SKU.
- [ ] **Focus Sharpness**: Center stone facets, prong tips, and shank outlines are crisp; Laplacian variance $\ge 100.0$.
- [ ] **Exposure & Glare Check**: No facet burnout clipping; mean background luminance is clean; stone body color is unblemished.
- [ ] **Staging Hygiene**: No dust fibers, grease marks, fingerprints, or visible mounting putty in the frame.
- [ ] **Centering & Alignment**: Ring is vertically centered with $\ge 10\%$ safety margin from image boundaries.
- [ ] **Provenance Verification**: Physical barcode on ring tag matches the `sku_id` and ERP lineage sheet.

---

## 6. Manifest Template (CSV Format)

```csv
asset_id,sku_id,identity_group_id,category,view_angle,capture_session_id,metal_color,metal_purity,gemstone_type,gemstone_shape,setting_type,rel_path,sha256
RING_SKU001_VIEW01,SKU-RING-001,PHYS-RING-001,ring,crown,SESS_01,white_gold,18k,diamond,round_brilliant,solitaire_prong,Jewellery_Data/ring/SKU-RING-001_01.jpg,CHECKSUM1
RING_SKU001_VIEW02,SKU-RING-001,PHYS-RING-001,ring,angle_45,SESS_01,white_gold,18k,diamond,round_brilliant,solitaire_prong,Jewellery_Data/ring/SKU-RING-001_02.jpg,CHECKSUM2
RING_SKU001_VIEW03,SKU-RING-001,PHYS-RING-001,ring,profile,SESS_01,white_gold,18k,diamond,round_brilliant,solitaire_prong,Jewellery_Data/ring/SKU-RING-001_03.jpg,CHECKSUM3
RING_SKU001_VIEW04,SKU-RING-001,PHYS-RING-001,ring,shank,SESS_01,white_gold,18k,diamond,round_brilliant,solitaire_prong,Jewellery_Data/ring/SKU-RING-001_04.jpg,CHECKSUM4
```
