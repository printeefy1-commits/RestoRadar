# Synthetic video fixture

`trial-preview.mp4` contains only FFmpeg's generated test pattern, without people or user footage.
It tests browser MP4 playback, seeking, mapping and overlays. It does not test AI accuracy.

Generated with:

```bash
ffmpeg -f lavfi -i testsrc2=size=320x240:rate=10 -t 8 -an \
  -c:v libx264 -preset veryslow -crf 38 -pix_fmt yuv420p \
  -movflags +faststart trial-preview.mp4
```
