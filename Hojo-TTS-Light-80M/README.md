[![github license](https://img.shields.io/github/license/HojoAI/Hojo-TTS-Light)]

## Hojo-TTS-Light-80M

## Demo

**Code-switching Male Voice**

https://github.com/user-attachments/assets/ad98ad4e-180b-48a7-9c0d-9745eaf869b4

**English Female Voice**

https://github.com/user-attachments/assets/c5cc087f-5b58-4556-93ad-bab160f30d53

**Chinese Female Voice (Sample 1)**

https://github.com/user-attachments/assets/aa9843e9-5c37-4a80-ab37-eafa61edf7ef

**English Male Voice**

https://github.com/user-attachments/assets/46e510dc-d4b6-4076-af11-849ee83884e5

**Chinese Female Voice (Sample 2)**

https://github.com/user-attachments/assets/8501450e-4942-4eea-a8d9-4923d42a7bba

## Environment Configuration & Inference Guide

### Environment Setup

```bash
git clone https://github.com/HojoAI/Hojo-TTS-Light.git
cd Hojo-TTS-Light
# Create a conda environment named hojo-tts with Python 3.12
conda create -n hojo-tts python=3.12 -y

# Activate the conda environment
conda activate hojo-tts

# Install inference dependencies
pip install -r requirements.txt
```

### Download ONNX Models from HuggingFace

```bash
pip install -U "huggingface_hub[cli]"
hf download HojoAI/Hojo-TTS-Light   --repo-type model   --local-dir ./models
```

### Run Inference

```bash
python infer.py \
  --onnx_dir ./models \
  --prompt_speech ./assets/zh1.wav \
  --prompt_text "现在的外卖确实坑多，要不咱换家稍微贵点的？可能品质好点。" \
  --text "今天天气怎么样。" \
  --output_path ./assets/out.wav
```

## Commercial Support

We offer commercial support for teams integrating Hojo TTS into their products. This includes integration assistance, custom voice development, and enterprise licensing.

Contact us or email developer@hojoai.com to discuss your requirements.

## Credits

- [X-Codec-2.0](https://github.com/zhenye234/X-Codec-2.0)
- [soprano](https://github.com/ekwek1/soprano)
- [inworld-ai/tts](https://github.com/inworld-ai/tts)

## Licence

This project is open-sourced under the [Apache 2.0 License](LICENSE.txt), which can be freely used for academic research, personal projects, and commercial secondary development.
