
[![github license](https://img.shields.io/github/license/HojoAI/Hojo-TTS-Light)]


## Hojo-TTS-Light
**Hojo-TTS-Light** is an open-source lightweight Text-To-Speech model by HojoAI team.</font>
There are currently two types of model parameter sizes, **Hojo-TTS-Light-80M** and **Hojo-TTS-Light-40M**.
Both follow the **token-LM** framework and both can generate good enough quality speech (average **DNSMOS>4.0** on Seed-TTS eval dataset).
Currently, Hojo-TTS-Light supports both Chinese and English. 
**Hojo-TTS-Light-40M** supports 15 build-in voices，2 for Chinese and 13 for English.
**Hojo-TTS-Light-80M** also supports voice cloning with a few seconds of audio.</font> 
We provided optimized ONNX model packaging file, so that users can perform speech synthesis efficiently on CPU without requiring a GPU.


## Roadmap
- [x] release Hojo-TTS-Light v1 Hojo-TTS-Light-80M ONNX model and inference engine
- [x] release Hojo-TTS-Light-40M ONNX model and inference engine
- [x] Hojo-TTS-Light-80M has been updated to the optimized ONNX model file（KV cache merging)
- [ ] support streaming mode synthesis
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
