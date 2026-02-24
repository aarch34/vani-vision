# Model Directory — Vāṇī-Vision

Place your downloaded GGUF model file here:

**Required file:**
```
models/phi-3-mini-4k-instruct-q4.gguf
```

## Download Instructions

1. Visit: https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf
2. Download `Phi-3-mini-4k-instruct-q4.gguf`  (~2.3 GB)
3. Place it in this `models/` directory

Or use huggingface-hub CLI:
```bash
pip install huggingface-hub
huggingface-cli download microsoft/Phi-3-mini-4k-instruct-gguf Phi-3-mini-4k-instruct-q4.gguf --local-dir models/
```

## Alternative Lightweight Models

If storage is limited, try:
- `Phi-3-mini-4k-instruct-q2_K.gguf` (~1.3 GB, faster, slightly less accurate)
- `TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf` (~0.6 GB, for very low RAM devices)

Update `config.py → LLM_MODEL_PATH` to match your downloaded file name.

## Note

The app runs in **demo mode** (with canned Socratic responses) if no model file
is present — useful for UI demonstration without downloading the full model.
