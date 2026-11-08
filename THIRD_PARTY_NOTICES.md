# Third-party tools and rights

This repository contains original Python/Swift source and original synthetic test data. It does not vendor third-party source, binaries, models, Apple DTDs, stock footage or user media. The MIT license applies only to material the project distributes under that license, not to separately installed software or user inputs.

- **Python**: the command-line code uses Python's standard library, distributed separately under the [Python Software Foundation license](https://docs.python.org/3/license.html).
- **FFmpeg**: an optional, separately installed command-line dependency for extracting frames and generating synthetic fixtures. FFmpeg is primarily LGPL 2.1-or-later; enabled components can make a particular build GPL. See the [official legal page](https://ffmpeg.org/legal.html) and the license/configuration of your installed build. This repository does not redistribute FFmpeg.
- **Swift / Apple Vision / Final Cut Pro**: Swift source calls the installed Apple Vision framework. Apple's SDKs, frameworks, application and installed FCPXML DTDs remain subject to Apple's terms. None is redistributed. See [Apple developer agreements](https://developer.apple.com/support/terms/) and [Final Cut Pro](https://www.apple.com/final-cut-pro/).
- **Whisper**: the tool accepts preexisting local transcript CSVs; it does not include or execute Whisper or model weights. If you use OpenAI Whisper or MLX Whisper to prepare input, consult their own repositories and licenses: [OpenAI Whisper](https://github.com/openai/whisper), [MLX Examples](https://github.com/ml-explore/mlx-examples).

Synthetic fixtures are generated from simple colors, shapes, tones and invented labels. They are test data, not recordings of real people or projects. Contributors must not add real media, transcripts, credentials, proprietary DTDs or third-party assets without an appropriate license and explicit publication permission.

Final Cut Pro and Apple Vision are Apple trademarks. This project is independent and is not endorsed by Apple.
