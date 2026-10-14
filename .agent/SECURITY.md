# Security & Data-Handling Policy

## 1. Input Validation & Upload Restrictions
- **MIME & Magic Byte Verification**: Uploaded images are validated using Pillow (`Image.open()` and `verify()`). Non-image files or malicious payloads disguised as images are rejected with HTTP 400.
- **File Size Cap**: Uploads are restricted to 10MB to prevent denial-of-service via memory exhaustion.
- **Path Traversal Protection**: All stored assets, uploads, and feedback files are sanitized. No user-supplied filenames are directly written to disk or accessed via relative paths.

## 2. Information Disclosure Prevention
- Local filesystem paths (e.g. `/Users/amol/...`) are stripped from public API responses. Images are referenced by relative identifiers or served through controlled media endpoints (`/media/...`).
- No external URL fetching: Image uploads must be multipart form uploads directly from the client. No remote URL scraping/fetching (SSRF prevention).

## 3. Storage & Privacy
- User-uploaded query images for search are processed in memory and only temporarily written for processing if needed; queries are not persisted to disk unless explicit user feedback is submitted.
- User feedback (`/api/feedback`) stores only the anonymous query hash, selected item ID, and timestamp.
