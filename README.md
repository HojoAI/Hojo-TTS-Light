[![github license](https://img.shields.io/github/license/HojoAI/Hojo-TTS-Light)]

## Hojo-TTS-Light

**Hojo-TTS-Light** is an open-source lightweight Text-To-Speech model by HojoAI team.

There are currently two types of model parameter sizes, **Hojo-TTS-Light-80M** and **Hojo-TTS-Light-40M**.
Both follow the **token-LM** framework and both can generate good enough quality speech (average **DNSMOS>4.0** on Seed-TTS eval dataset).

Currently, Hojo-TTS-Light supports both Chinese and English.

**Hojo-TTS-Light-40M** supports 15 build-in voices，2 for Chinese and 13 for English.
**Hojo-TTS-Light-80M** also supports voice cloning with a few seconds of audio.
We provided optimized ONNX model packaging file, so that users can perform speech synthesis efficiently on CPU without requiring a GPU.

## Features

- **Ultra-Lightweight Core Model** --- The core language model is only 80M and 40M parameters, with extremely small parameter size under the same sound quality and very low deployment threshold.
- **Native Bilingual Integration** --- A single model supports smooth synthesis for both Chinese and English, no branch switching required.
- **Low Computational Cost & On-Device Friendly** --- Low memory usage and low inference overhead, which can run smoothly on CPU, ordinary GPU, and embedded edge devices.
- **Ready to Use** --- Provides simple inference scripts and fast calling interfaces, enabling synthesis and cloning with one line of code, facilitating secondary development and business integration.
- **Supports quick correction** --- For the problem of easily mispronouncing Chinese and English polyphonic characters and proper nouns, users can directly use Pinyin to correct pronunciation errors and improve the reliability of speech synthesis.

## Model Details

- The model follows the Token-LM model paradim.
- The speech tokenizer is composed of a **18M** encoder and a **30M** decoder.
- Currently the released version runs at **50Hz** token rate and the **12.5hz** version models will be released soon.

## Roadmap
- [ ] support streaming mode synthesis
- [ ] support emotion and style control
- [ ] support multi-lingual and multi-dialect

- [X] 202608
	- [X] release Hojo-TTS-Light v2 Hojo-TTS-Light-80M ONNX model and inference engine, add fine local model.
	- [X] release Hojo-TTS-Light v2 Hojo-TTS-Light-40M ONNX model and inference engine, add fine local model.
- [X] 202606
	- [X] Hojo-TTS-Light has been updated to the optimized ONNX model file (KV cache merging)
	- [X] release Hojo-TTS-Light-40M ONNX model and inference engine
- [X] 202605
	- [X] release Hojo-TTS-Light v1 Hojo-TTS-Light-80M ONNX model and inference engine

## Commercial Support

We offer commercial support for teams integrating Hojo TTS into their products. This includes integration assistance, custom voice development, and enterprise licensing.

Contact us or email developer@hojoai.com to discuss your requirements.

## Star History

<a href="https://www.star-history.com/?repos=HojoAI%2FHojo-TTS-Light&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=HojoAI/Hojo-TTS-Light&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=HojoAI/Hojo-TTS-Light&type=date&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=HojoAI/Hojo-TTS-Light&type=date&legend=top-left" />
 </picture>
</a>

## Credits

- [X-Codec-2.0](https://github.com/zhenye234/X-Codec-2.0)
- [soprano](https://github.com/ekwek1/soprano)
- [inworld-ai/tts](https://github.com/inworld-ai/tts)

## Licence

This project is open-sourced under the [Apache 2.0 License](LICENSE.txt), which can be freely used for academic research, personal projects, and commercial secondary development.
