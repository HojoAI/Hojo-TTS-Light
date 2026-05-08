
![License](http://www.apache.org/licenses/LICENSE-2.0)

## Hojo-TTS-Light
**Hojo-TTS-Light** is an open-source lightweight Text-To-Speech model by HojoAI team.
**With only 0.08B parameters**, that is, the parametere size of backbone LM is only **80M**, Hojo-TTS-Light can generate good enough quality speech.
Currently, Hojo-TTS-Light supports both Chinese and English, and also supports voice cloning with as little as a few seconds of audio.
## Features
- **Ultra-Lightweight Core Model** --- The core language model is only 80M parameters, with extremely small parameter size under the same sound quality and very low deployment threshold.
- **Native Bilingual Integration** --- A single model supports smooth synthesis and cross-lingual voice cloning for both Chinese and English, no branch switching required.
- **Few-Shot Voice Cloning** --- High similarity voice cloning can be completed with a small amount of reference audio, featuring natural prosody, high voice restoration.
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


## Quick Start
**TBD**
## Credits
- [X-Codec-2.0](https://github.com/zhenye234/X-Codec-2.0)
- [soprano](https://github.com/ekwek1/soprano)
- [inworld-ai/tts](https://github.com/inworld-ai/tts)
## Licence
This project is open-sourced under the [Apache 2.0 License](LICENSE.txt), which can be freely used for academic research, personal projects, and commercial secondary development.
