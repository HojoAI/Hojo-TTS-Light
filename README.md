
![github license](https://img.shields.io/github/license/HojoAI/Hojo-TTS-Light)


## Hojo-TTS-Light
**Hojo-TTS-Light** is an open-source lightweight Text-To-Speech model by HojoAI team.
**With only 0.08B parameters**, that is, the parametere size of backbone LM is only **80M**, Hojo-TTS-Light can generate good enough quality speech (average **DNSMOS>4.0** on Seed-TTS eval dataset).
Currently, Hojo-TTS-Light supports both Chinese and English, and also supports voice cloning with a few seconds of audio.
## Features
- **Ultra-Lightweight Core Model** --- <font size=3>The core language model is only 80M parameters, with extremely small parameter size under the same sound quality and very low deployment threshold.</font>
- **Native Bilingual Integration** --- <font size=3>A single model supports smooth synthesis and cross-lingual voice cloning for both Chinese and English, no branch switching required.</font>
- **Voice Cloning** --- <font size=3>High similarity voice cloning can be completed with a small amount of reference audio, featuring natural prosody, high voice restoration.</font>
- **Low Computational Cost & On-Device Friendly** --- <font size=3>Low memory usage and low inference overhead, which can run smoothly on CPU, ordinary GPU, and embedded edge devices.</font>
- **Ready to Use** --- <font size=3>Provides simple inference scripts and fast calling interfaces, enabling synthesis and cloning with one line of code, facilitating secondary development and business integration.</font>
- **Supports quick correction** --- <font size=3>For the problem of easily mispronouncing Chinese and English polyphonic characters and proper nouns, users can directly use Pinyin to correct pronunciation errors.</font>

## Model Details
- The model follows the Token-LM model paradim.
- The speech tokenizer is composed of a **18M** encoder and a **30M** decoder.
- We use FSQ which inherently enables higher codebook utilization, the codebook size is 8000 for audio and totally <20000.
- Currently the released version runs at **50Hz** token rate and the **12.5hz** version models will be released soon.
  
## Demo
**Code-switching Male Voice**
[mixed.mp3](https://github.com/user-attachments/files/27513409/mix_13.mp3)

**English Female Voice**
[female_en.mp3](https://github.com/user-attachments/files/27513300/female_en_139.mp3)

**Chinese Female Voice (Sample 1)**
[female_zh_1.mp3](https://github.com/user-attachments/files/27513312/female_zh_89.mp3)

**English Male Voice**
[male_en.mp3](https://github.com/user-attachments/files/27513325/male_en_88.mp3)

**Chinese Female Voice (Sample 2)**
[female_zh_2.mp3](https://github.com/user-attachments/files/27513335/female_zh_95.mp3)

## Environment Configuration ; Inference Guide for Hojo\-TTS

###  Environment Setup

```bash
git clone https://github.com/HojoAI/Hojo-TTS-Light.git
cd Hojo-TTS-Light
# Create a conda environment named hojo-tts with Python 3.12
conda create -n hojo-tts python=3.12 -y

# Activate the conda environment
conda activate hojo-tts

# Install inference dependencies
pip install -r requirements.infer.txt
```

### Download ONNX Models from HuggingFace

```bash
pip install -U "huggingface_hub[cli]"
hf download HojoAI/Hojo-TTS-Light   --repo-type model   --include "onnx/*"   --local-dir .
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

## Roadmap
- [ ] support streaming mode synthesis, release optimized inference engine
- [ ] support emotion and style control
- [ ] support multi-lingual and multi-dialect

  
## Commercial Support
We offer commercial support for teams integrating Hojo TTS into their products. This includes integration assistance, custom voice development, and enterprise licensing.

Contact us or email developer@hojoai.com to discuss your requirements.
  
## Credits
- [X-Codec-2.0](https://github.com/zhenye234/X-Codec-2.0)
- [soprano](https://github.com/ekwek1/soprano)
- [inworld-ai/tts](https://github.com/inworld-ai/tts)
## Licence
This project is open-sourced under the [Apache 2.0 License](LICENSE.txt), which can be freely used for academic research, personal projects, and commercial secondary development.
