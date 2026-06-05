# JachAI Postman Bundle

This folder contains a ready-to-import Postman collection and environment for the JachAI backend.

## Files

- `JachAI_Backend.postman_collection.json`
- `JachAI_Backend.local.postman_environment.json`
- `test-materials/`

## Import

1. Open Postman.
2. Import both JSON files from this folder.
3. Select the `JachAI Backend Local` environment.
4. Set `internal_api_key` to the same value used by the backend `.env`.
5. Keep the `test-materials/` folder in place if you want the bundled image file path to remain valid.

## Recommended Run Order

1. `Health`
2. `Sources`
3. `Claims`
4. `Dashboard`
5. `Rumor Clusters`
6. `Webhooks`
7. `Negative Cases`

## Notes

- The collection automatically stores `claim_id`, `job_id`, `source_id`, and `cluster_id` as collection variables after successful requests.
- `Sources > 01 Ingest Sample Sources` should be run before the main claim tests so retrieval has evidence to work with.
- Sample request materials live in [test-materials/README.md](/run/media/bakau/WORKPLACE/JachAI/backend/postman/test-materials/README.md:1).
- `Claims > 02 Submit Image Claim` now ships with a bundled image at [sample_claim_image.png](/run/media/bakau/WORKPLACE/JachAI/backend/postman/test-materials/sample_claim_image.png:1). If Postman does not auto-attach it after import, re-select that file manually.
- The expected OCR text for the bundled image is in [sample_claim_image_expected_ocr.txt](/run/media/bakau/WORKPLACE/JachAI/backend/postman/test-materials/sample_claim_image_expected_ocr.txt:1).
- Image OCR depends on Tesseract being installed on the backend host.
- URL-based claim and webhook requests depend on outbound internet access from the backend host.

## Optional Newman Run

```bash
newman run postman/JachAI_Backend.postman_collection.json \
  -e postman/JachAI_Backend.local.postman_environment.json
```
