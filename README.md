# video-descriptor

`video-descriptor` extracts audio from a video with `ffmpeg` and transcribes it with WhisperX. It is designed to run in Google Colab with GPU acceleration and generate transcript outputs in `JSON`, `TXT`, `SRT`, and `VTT`.

[Open the Colab notebook](https://colab.research.google.com/github/pballestawork/video-descriptor/blob/main/notebooks/video_descriptor_colab.ipynb)

## Colab workflow

1. Open `notebooks/video_descriptor_colab.ipynb` in Colab.
2. Select a GPU runtime: `Runtime > Change runtime type > Hardware accelerator > GPU`.
3. Run the install cell. The first run installs PyTorch/WhisperX and restarts the runtime.
4. Run the same install cell again after the restart; it should print the installed versions and `Entorno listo.`
5. Upload a video when prompted.
6. Run the transcription cell.
7. Download the generated ZIP with transcript files.

The default model is `large-v3-turbo`, the default language is Spanish (`es`), and the Colab notebook enables diarization for two speakers by default.

## Large video note

The notebook uses `google.colab.files.upload()` as the primary input path because it is simple for interactive use. Colab does not publish a stable file-size limit for this browser upload path, and large uploads can fail because of browser/network interruptions, runtime disconnects, or temporary disk limits in `/content`.

For long videos, the safer workflow is:

1. Upload the video to Google Drive.
2. Mount Drive in Colab.
3. Copy the video into `/content` before processing.

```python
from google.colab import drive
drive.mount("/content/drive")

!cp "/content/drive/MyDrive/path/to/video.mp4" "/content/video.mp4"
VIDEO_PATH = "/content/video.mp4"
```

Google Drive documents much higher storage limits, including files up to 5 TB and upload/copy constraints such as 750 GB/day in some Workspace/API contexts. See the Drive API limits and Colab FAQ for current details.

## Local usage

Install the package locally:

```bash
python -m pip install -e .
python -m pip install -r requirements-colab.txt
```

In Colab, use the notebook install cell. The project pins `whisperx==3.3.0`, `torch/torchaudio==2.8.0`, `numpy==2.0.2`, `pandas==2.2.2` and compatible `opentelemetry` packages. Current WhisperX releases can upgrade Colab to `numpy>=2.1`, `pandas>=3` and newer telemetry packages. Also, `pyannote.audio==3.3.2` needs `torchaudio.AudioMetaData`, which exists in `torchaudio==2.8.0` but was removed in `torchaudio>=2.9`. The install cell intentionally restarts the runtime after installing these packages; run it once to install/restart and a second time to verify. The warning below from `apt-get update` is common in Colab and is not fatal:

```text
W: Skipping acquire of configured file 'main/source/Sources' ...
```

The first install can take several minutes because PyTorch CUDA wheels are large. The notebook prints each install step, elapsed time and `pip` download progress so you can see that Colab is still working.

If you stop the install cell manually, Colab will show a `KeyboardInterrupt`. That only means the running `pip` process was cancelled. The notebook does not mark the install as complete until every step finishes, so the cleanest recovery is to delete the runtime and run the install cell again.

PyTorch 2.6+ changed `torch.load` to use `weights_only=True` by default. WhisperX 3.3.0 loads a trusted pyannote VAD checkpoint that still needs the previous behavior, so the CLI forces `weights_only=False` inside the isolated transcription process before loading WhisperX models.

After the heavy install has completed once, the install cell still force-updates only the lightweight `video-descriptor` package on later runs. This keeps Colab on the latest project code without reinstalling PyTorch.

Run a transcription:

```bash
python -m video_descriptor transcribe INPUT_VIDEO \
  --output-dir outputs \
  --model large-v3-turbo \
  --language es \
  --batch-size 16 \
  --compute-type float16 \
  --device auto
```

To identify two interlocutors as `Persona A` and `Persona B`, enable diarization:

```bash
export HF_TOKEN="your_huggingface_token"
python -m video_descriptor transcribe INPUT_VIDEO \
  --output-dir outputs \
  --language es \
  --diarize \
  --num-speakers 2
```

Diarization uses pyannote through WhisperX and requires a Hugging Face token with access to the pyannote diarization model terms accepted. WhisperX assigns voice clusters, not real names, so the project maps speakers to `Persona A`, `Persona B`, etc.

When `--device auto` is used, the tool selects CUDA when available. If it falls back to CPU, it uses `int8` compute type to avoid unsupported/expensive `float16` CPU execution.

## Outputs

For an input named `meeting.mp4`, the output directory will contain:

- `meeting.wav`: extracted mono 16 kHz audio
- `meeting.json`: WhisperX structured result
- `meeting.txt`: readable transcript with segment timestamps
- `meeting.srt`: subtitles
- `meeting.vtt`: web subtitles

## References

- WhisperX: https://github.com/m-bain/whisperX
- Colab FAQ: https://research.google.com/colaboratory/intl/es/faq.html
- Drive API limits: https://developers.google.com/workspace/drive/api/guides/limits
