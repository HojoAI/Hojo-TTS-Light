
![github license](https://img.shields.io/github/license/HojoAI/Hojo-TTS-Light)


## Hojo-TTS-Light
**Hojo-TTS-Light** is an open-source lightweight Text-To-Speech model by HojoAI team.
**With only 0.08B parameters**, that is, the parametere size of backbone LM is only **80M**, Hojo-TTS-Light can generate good enough quality speech (average **DNSMOS>4.0** on Seed-TTS eval dataset).
Currently, Hojo-TTS-Light supports both Chinese and English, and also supports voice cloning with a few seconds of audio.
## Features
- **Ultra-Lightweight Core Model** --- The core language model is only 80M parameters, with extremely small parameter size under the same sound quality and very low deployment threshold.
- **Native Bilingual Integration** --- A single model supports smooth synthesis and cross-lingual voice cloning for both Chinese and English, no branch switching required.
- **Voice Cloning** --- High similarity voice cloning can be completed with a small amount of reference audio, featuring natural prosody, high voice restoration.
- **Low Computational Cost & On-Device Friendly** --- Low memory usage and low inference overhead, which can run smoothly on CPU, ordinary GPU, and embedded edge devices.
- **Ready to Use** --- Provides simple inference scripts and fast calling interfaces, enabling synthesis and cloning with one line of code, facilitating secondary development and business integration.
- **Supports quick correction** --- For the problem of easily mispronouncing Chinese and English polyphonic characters and proper nouns, users can directly use Pinyin to correct pronunciation errors.

## Model Details
- The model follows the Token-LM model paradim.
- The speech tokenizer is composed of a **18M** encoder and a **30M** decoder.
- We use FSQ which inherently enables higher codebook utilization, the codebook size is 8000 for audio and totally <20000.
- Currently the released version runs at **50Hz** token rate and the **12.5hz** version models will be released soon.
  
## Demo

### English Female Voice
<audio controls>
  <source src="https://github.com/user-attachments/files/27511331/female_en_139.mp3" type="audio/mpeg">
  Your browser does not support the audio element.
</audio>

### Chinese Female Voice (Sample 1)
<audio controls>
  <source src="https://github.com/user-attachments/files/27511327/female_zh_95.mp3" type="audio/mpeg">
  Your browser does not support the audio element.
</audio>

### English Male Voice
<audio controls>
  <source src="https://github.com/user-attachments/files/27511323/male_en_88.mp3" type="audio/mpeg">
  Your browser does not support the audio element.
</audio>

### Chinese Female Voice (Sample 2)
<audio controls>
  <source src="https://github.com/user-attachments/files/27511322/female_zh_89.mp3" type="audio/mpeg">
  Your browser does not support the audio element.
</audio>

## Environment Configuration \&amp; Inference Guide for Hojo\-TTS

###  Environment Setup

```bash
# Create a conda environment named hojo-tts with Python 3.12
conda create -n hojo-tts python=3.12 -y

# Activate the conda environment
conda activate hojo-tts

# Install inference dependencies
pip install -r requirements.infer.txt
```

### Download ONNX Models from HuggingFace

#### Install the Download Tool

```bash
pip install -U "huggingface_hub[cli]"
```

####  Download the Models

Replace \&lt;HF\_MODEL\_REPO\_URL\&gt; with your actual HuggingFace model repository URL:

```bash
huggingface-cli download <HF_MODEL_REPO_URL> \
  --local-dir ./onnx \
  --local-dir-use-symlinks False
```

####  Verify Downloaded Files

After downloading, ensure the following files exist in the `onnx/` directory:

- onnx/Hojo\-TTS\-Light\-llm\.onnx

- onnx/Hojo\-TTS\-Light\-encoder\.onnx

- onnx/Hojo\-TTS\-Light\-decoder\.onnx

###  Run Inference

```bash
python infer_onnx.py \
  --onnx_dir ./onnx \
  --tokenizer_path ./tokenizer \
  --prompt-speech ./assets/zh1.wav \
  --prompt-text "现在的外卖确实坑多，要不咱换家稍微贵点的？可能品质好点。" \
  --text "今天天气怎么样。" \
  --output-wav ./assets/out.wav
```


  
## Credits
- [X-Codec-2.0](https://github.com/zhenye234/X-Codec-2.0)
- [soprano](https://github.com/ekwek1/soprano)
- [inworld-ai/tts](https://github.com/inworld-ai/tts)
## Licence
This project is open-sourced under the [Apache 2.0 License](LICENSE.txt), which can be freely used for academic research, personal projects, and commercial secondary development.
