# Postman Test Materials

This folder contains reusable test assets for the JachAI Postman collection.

## Files

- `source_ingest_payload.json`
  Used by `Sources > 01 Ingest Sample Sources`
- `claim_text_payload.json`
  Example body for `Claims > 01 Submit Text Claim`
- `claim_url_payload.json`
  Example body for `Claims > 03 Submit URL Claim`
- `review_status_payload.json`
  Example body for `Claims > 07 Update Claim Review Status`
- `webhook_text_payload.json`
  Example body for `Webhooks > 01 Inbound Message Webhook - Text`
- `webhook_url_payload.json`
  Example body for `Webhooks > 02 Inbound Message Webhook - URL`
- `unauthorized_source_ingest_payload.json`
  Negative-case body for `Negative Cases > 01 Unauthorized Source Ingest`
- `invalid_text_claim_payload.json`
  Negative-case body for `Negative Cases > 02 Invalid Text Claim`
- `invalid_webhook_payload.json`
  Negative-case body for `Negative Cases > 04 Invalid Webhook Payload`
- `sample_claim_image.png`
  OCR test image for `Claims > 02 Submit Image Claim`
- `sample_claim_image_expected_ocr.txt`
  Reference text expected from the OCR image

## Notes

- The JSON files are plain request-body materials you can open, inspect, or reuse in other tools.
- `sample_claim_image.png` is a high-contrast English sample intended to be easy for Tesseract OCR.
- If you move the repository, update the image file path in Postman if needed.
