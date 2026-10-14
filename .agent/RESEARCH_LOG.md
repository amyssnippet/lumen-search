# Research Log

## Entry 1: 2026-10-11 — Dataset Distribution and Representation Nuances
- **Question**: What are the visual properties and structural challenges of the raw jewellery dataset?
- **Investigation**: Audited all 492 images using Pillow and difference hashing.
- **Findings**:
  - 429 of 492 images are exactly 262x262 thumbnails.
  - No corrupted images.
  - Rings feature diverse cuts (round brilliant, oval, emerald, marquise), metal colors (yellow gold, white gold/platinum, rose gold), and settings (solitaire, halo, 3-stone, pavé band).
  - Necklaces feature diverse chain types (rope, box, cable) and pendants (medallions, crosses, solitaires, teardrops).
  - Several items are identical piece photographs with different levels of JPEG compression or cropped margins (84 near-duplicate pairs with dHash $\le 2$).
- **Implication**: Models trained on coarse ImageNet classes struggle to distinguish prongs and gemstone facet counts without self-supervised patch tokens. DINOv2 is an optimal candidate because its self-supervised objective attends directly to fine geometric borders and surface textures.

## Entry 2: 2026-10-11 — Vision Transformer Latency on Intel CPU
- **Question**: Can a Vision Transformer run interactively on Intel Mac CPU?
- **Investigation**: Tested `vit_small_patch14_dinov2` with default 518x518 vs dynamic 224x224.
- **Findings**: Default 518x518 requires 1.54s per image on CPU; dynamic 224x224 patch interpolation drops forward latency to 306ms per image.
- **Implication**: For interactive search, 224x224 dynamic resolution provides a 5x speedup with minimal loss in high-level jewellery semantic discrimination.
