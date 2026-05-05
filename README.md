# video-descriptor

`video-descriptor` extracts audio from a video with `ffmpeg` and transcribes it with WhisperX. It is designed to run in Google Colab with GPU acceleration and generate transcript outputs in `JSON`, `TXT`, `SRT`, and `VTT`.

[Open the Colab notebook](https://colab.research.google.com/github/pballestawork/video-descriptor/blob/main/notebooks/video_descriptor_colab.ipynb)

## Colab workflow

1. Open `notebooks/video_descriptor_colab.ipynb` in Colab.
2. Select a GPU runtime: `Runtime > Change runtime type > Hardware accelerator > GPU`.
3. Run the install cell.
4. Upload a video when prompted.
5. Run the transcription cell.
6. Download the generated ZIP with transcript files.

The default model is `large-v3-turbo`, the default language is Spanish (`es`), and diarization is intentionally disabled in v1.

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
