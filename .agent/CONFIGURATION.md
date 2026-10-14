# Configuration Reference

The application uses standard environment variables and sensible defaults configured via Pydantic or CLI flags.

| Variable / Setting | Default Value | Description |
|---|---|---|
| `MODEL_NAME` | `vit_small_patch14_dinov2` | Backbone vision encoder (`vit_small_patch14_dinov2`, `ViT-B-32-clip`) |
| `IMAGE_SIZE` | `224` | Input resolution for model feature extraction |
| `EMBEDDING_DIM` | `384` | Dimension of the normalized embedding vectors (384 for DINOv2 small, 512 for CLIP) |
| `INDEX_DIR` | `data/index` | Directory for serialized numpy embeddings and catalog metadata |
| `DATA_DIR` | `Jewellery_Data` | Root directory containing catalog image assets |
| `EXACT_MATCH_THRESHOLD` | `0.94` | Calibrated cosine similarity threshold for verified exact SKU identity |
| `SIMILARITY_THRESHOLD` | `0.70` | Threshold below which items are rejected as non-matches |
| `MAX_UPLOAD_SIZE_MB` | `10` | Maximum file upload size accepted by search API |
| `ALLOWED_EXTENSIONS` | `.jpg,.jpeg,.png,.webp` | Allowed image MIME types and file extensions |
| `HOST` | `0.0.0.0` | API bind address |
| `PORT` | `8000` | API server port |
