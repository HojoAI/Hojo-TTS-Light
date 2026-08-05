# Hojo TTS Light 40M (ONNX, 24 kHz)

Lightweight **speaker-conditioned text-to-speech** with **15 preset voices**. Run with Python + ONNX Runtime — no PyTorch.

## Demo

https://github.com/user-attachments/assets/79023517-80e3-4d67-837e-c4b4f9367a0b

**Hugging Face:** [`HojoAI/Hojo-TTS-Light-40M`](https://huggingface.co/HojoAI/Hojo-TTS-Light-40M)

- **24 kHz** mono output
- **15 voices** via opaque IDs in `Hojo-TTS-Light-40M-voice.npz` (e.g. `hojo_en_m_02`)
- **CPU or GPU** (ONNX Runtime)
- **Up to 4096 LM tokens** per utterance (`max_position_embeddings` in `config.json`)
- Dependencies: `numpy`, `soundfile`, `tokenizers`, `onnxruntime`, `huggingface_hub` (for `--repo-id` / `hf download`)

## Quick start

```bash
pip install -r requirements.txt
```

### Get models

| Option | When to use |
|--------|-------------|
| **A** — local `models/` | Weights already in the repo checkout |
| **B** — `hf download` | Manual download to a chosen directory |
| **C** — `--repo-id` | Let `infer.py` fetch via `huggingface_hub` |

**A — bundled `models/`**

```bash
python infer.py \
  --text "Hello, this is a demo." \
  --voice hojo_en_m_02 \
  --output_path examples/outputs/demo.wav
```

**B — Hugging Face CLI**

```bash
pip install -U "huggingface_hub[cli]"
hf download HojoAI/Hojo-TTS-Light-40M --repo-type model --local-dir ./models

python infer.py \
  --onnx_dir ./models \
  --text "Hello, this is a demo." \
  --voice hojo_en_m_02 \
  --output_path examples/outputs/demo.wav
```

**C — `--repo-id`** (recommended for scripts)

```bash
python infer.py \
  --repo-id HojoAI/Hojo-TTS-Light-40M \
  --text "Hello, this is a demo." \
  --voice hojo_en_m_02 \
  --output_path examples/outputs/demo.wav
```

### Python API

```python
from infer import get_model

HF_REPO = "HojoAI/Hojo-TTS-Light-40M"

tts = get_model()  # default: ./models/
# tts = get_model(repo_id=HF_REPO)

wav = tts.generate("Hello.", voice="hojo_en_m_02")
tts.generate_to_file("Hello.", "out.wav", voice="hojo_en_m_02")
```

## Voices

Use `--voice` with one of the opaque IDs below (`hojo_{lang}_{sex}_{nn}`; `sex` is `f` / `m` in the ID, `female` / `male` in the table):

| Voice ID | Language | Sex |
|----------|----------|-----|
| `hojo_zh_f_01` | zh | female |
| `hojo_zh_f_02` | zh | female |
| `hojo_en_f_01` | en | female |
| `hojo_en_f_02` | en | female |
| `hojo_en_f_03` | en | female |
| `hojo_en_f_04` | en | female |
| `hojo_en_f_05` | en | female |
| `hojo_en_f_06` | en | female |
| `hojo_en_f_07` | en | female |
| `hojo_en_f_08` | en | female |
| `hojo_en_m_01` | en | male |
| `hojo_en_m_02` | en | male |
| `hojo_en_m_03` | en | male |
| `hojo_en_m_04` | en | male |
| `hojo_en_m_05` | en | male |

Default voice: **`hojo_en_m_02`**.

## Model bundle (`models/`)

Downloaded from `HojoAI/Hojo-TTS-Light-40M` (or shipped under `models/`):

| File | Role |
|------|------|
| `Hojo-TTS-Light-40M-llm.onnx` | Unified LM body (`inputs_embeds` → logits + hidden + KV; BF16 on disk) |
| `Hojo-TTS-Light-40M-fine_local.onnx` | Coarse embeddings + hidden → tokens logits (BF16 on disk) |
| `Hojo-TTS-Light-40M-decoder.onnx` | tokens → spectrogram mag/phase (FP32) |
| `Hojo-TTS-Light-40M-voice.npz` | Voices IDs |
| `tokenizer.json` / `tokenizer_config.json` | Tokenizer |
| `config.json` | LM config (`num_hidden_layers`, `max_position_embeddings`) |

Tensor shapes, initializer counts, and architecture notes: [`models/README.md`](models/README.md).

## Model size

| Component | Params | ~M |
|-----------|--------|-----|
| `Hojo-TTS-Light-40M-llm.onnx` | 31,337,600 | 31.34 |
| `Hojo-TTS-Light-40M-fine_local.onnx` | 13,296,256 | 13.30 |
| `Hojo-TTS-Light-40M-decoder.onnx` | 30,910,594 | 30.91 |
| **Total** | **75,544,450** | **75.55** |

Bundle on disk: ~**236 MB** (weights + tokenizer + voice.npz).  

## Layout

```
.
├── infer.py           # CLI + HojoTTSLight API
├── onnx_model.py      # HojoTTSLightOnnx runtime
├── requirements.txt
├── models/            # ONNX weights & config (see models/README.md)
└── examples/
    └── outputs/       # optional demo WAV output
```

## CLI reference

| Flag | Default | Meaning |
|------|---------|---------|
| `--onnx_dir` | `./models` | Local model directory |
| `--repo-id` | — | e.g. `HojoAI/Hojo-TTS-Light-40M` — downloads bundle via `huggingface_hub` |
| `--cache-dir` | — | HF cache directory (with `--repo-id`) |
| `--voices_npz` | `<onnx_dir>/Hojo-TTS-Light-40M-voice.npz` | Override voice embeddings |
| `--text` | — | Input text (required) |
| `--voice` | — | Voice ID (required) |
| `--output_path` | — | Output `.wav` path (required) |

## Performance (reference, CPU)

Short English sentence (~6.3 s audio, `hojo_en_m_02`), Linux server:

| Metric | 1 thread | 20 threads |
|--------|----------|------------|
| RTF | ~0.71× | ~0.28× |
| Peak RSS | ~750 MB | ~750 MB |

Long-form (~27 s audio, ~1420 LM tokens): RTF ~1.02× / ~0.59×, Peak RSS ~1.2 GB / ~1.2 GB.

RTF = wall-clock time ÷ output audio duration (< 1 = faster than realtime). Set ORT threads via `HojoTTSLightOnnx(..., num_threads=n)` when constructing the runtime. LM / FineLocal BF16 weights are promoted to FP32 in memory at load (ORT CPU), so RSS is higher than on-disk size.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
