# Hojo TTS Light 40M (ONNX, 24 kHz)

Lightweight **speaker-conditioned text-to-speech** with **15 preset voices**. Run with Python + ONNX Runtime — no PyTorch.
**Demo

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
| `Hojo-TTS-Light-40M-llm.onnx` | Unified LM (prefill + decode, KV cache) |
| `Hojo-TTS-Light-40M-decoder.onnx` | Audio codec (tokens → spectrogram) |
| `Hojo-TTS-Light-40M-voice.npz` | Speaker embeddings for 15 voice IDs |
| `tokenizer.json` / `tokenizer_config.json` | Tokenizer |
| `config.json` | LM config (`num_hidden_layers`, `max_position_embeddings`) |

Tensor shapes, initializer counts, and architecture notes: [`models/README.md`](models/README.md).

## Model size

| Component | Params | ~M |
|-----------|--------|-----|
| `Hojo-TTS-Light-40M-llm.onnx` | 40,392,320 | 40.39 |
| `Hojo-TTS-Light-40M-decoder.onnx` | 32,797,906 | 32.80 |
| **Total** | **73,190,226** | **73.19** |

Bundle on disk: ~**220 MB** (weights + tokenizer + voices).

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

Short English sentence (~6.4 s audio, `hojo_en_m_02`), Linux server:

| Metric | 1 thread | 20 threads |
|--------|----------|------------|
| RTF | ~0.81× | ~0.46× |
| Peak RSS | ~410 MB | ~453 MB |

Long-form (~19–20 s audio, ~940 LM tokens): RTF ~0.92× / ~0.46×, Peak RSS ~512 MB / ~563 MB.

RTF = wall-clock time ÷ output audio duration (< 1 = faster than realtime). Set ORT threads via `onnx_model.configure_cpu_threads(n)` before loading the model.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
