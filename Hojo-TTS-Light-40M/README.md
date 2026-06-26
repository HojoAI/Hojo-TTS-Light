# Speaker-Conditioned TTS (ONNX, 24 kHz)


## Quick start

```bash
pip install -r requirements.txt
```

Hugging Face:

```bash
python infer.py \
  --repo-id your-org/Hojo-TTS-Light \
  --text "Hello, this is a demo." \
  --voice hojo_en_m_02 \
  --output_path examples/outputs/demo.wav
```

Python API:

```python
from infer import get_model

tts = get_model()  # local models/
wav = tts.generate("Hello.", voice="hojo_en_m_02")
```

## Voices

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

## Model size

| Component | Params | ~M |
|-----------|--------|-----|
| Unified LM (`Hojo-TTS-Light-40M-llm.onnx`) | 40,392,320 | 40.39 |
| Decoder (`Hojo-TTS-Light-40M-decoder.onnx`) | 32,797,906 | 32.80 |
| **Total** | **73,190,226** | **73.19** |


| Metric | 1 thread | 20 threads |
|--------|----------|------------|
| Output audio | 18.8 s @ 24 kHz | 18.8 s @ 24 kHz |
| Total latency | 17.3 s | 8.6 s |
| **RTF** | **0.92×** | **0.46×** |
| RSS after model load | ~391 MB | ~440 MB |
| **Peak RSS (process)** | **~512 MB** | **~563 MB** |

## License

Apache License 2.0 — see [LICENSE](LICENSE).
