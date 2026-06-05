# JachAI Postman Bundle

This folder contains a ready-to-import Postman collection and environment for the JachAI backend.

## Files

- `JachAI_Backend.postman_collection.json`
- `JachAI_Backend.local.postman_environment.json`

## Import

1. Open Postman.
2. Import both JSON files from this folder.
3. Select the `JachAI Backend Local` environment.
4. Set `internal_api_key` to the same value used by the backend `.env`.

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
- `Claims > 02 Submit Image Claim` requires you to choose a real local image file in Postman Desktop.
- Image OCR depends on Tesseract being installed on the backend host.
- URL-based claim and webhook requests depend on outbound internet access from the backend host.

## Optional Newman Run

```bash
newman run postman/JachAI_Backend.postman_collection.json \
  -e postman/JachAI_Backend.local.postman_environment.json
```
