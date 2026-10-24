"""Bounded, cancellable preparation of camera MP4s for browser playback."""

import asyncio
import subprocess
import time

from fastapi import HTTPException

CONVERSION_TIMEOUT_SECONDS = 600
MAX_CONVERSION_SECONDS = 120


async def prepare_browser_video(source, target, output_limit, request):
    # Leave room for the final MP4 header when ffmpeg stops at its byte limit.
    if output_limit <= 64 * 1024:
        raise HTTPException(507, "لا توجد مساحة كافية لتهيئة الفيديو")
    process = await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-protocol_whitelist",
        "file",
        "-threads",
        "2",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-map",
        "0:a:0?",
        "-sn",
        "-dn",
        "-filter_threads",
        "1",
        "-vf",
        "scale=w='min(1920,iw)':h='min(1080,ih)':force_original_aspect_ratio=decrease:force_divisible_by=2",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-threads",
        "2",
        "-fpsmax",
        "30",
        "-c:a",
        "aac",
        "-b:a",
        "96k",
        "-movflags",
        "+faststart",
        "-fs",
        str(output_limit - 64 * 1024),
        "-f",
        "mp4",
        str(target),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    started = time.monotonic()
    try:
        while process.returncode is None:
            if await request.is_disconnected():
                raise HTTPException(400, "أُلغي الرفع أثناء تهيئة الفيديو")
            if time.monotonic() - started > CONVERSION_TIMEOUT_SECONDS:
                raise HTTPException(
                    504, "استغرقت تهيئة الفيديو وقتًا طويلًا؛ جرّب مقطعًا أقصر"
                )
            await asyncio.sleep(0.2)
        if process.returncode != 0:
            raise HTTPException(
                422, "تعذرت تهيئة الفيديو تلقائيًا؛ تحقق من الملف أو جرّب مقطعًا أقصر"
            )
    finally:
        if process.returncode is None:
            process.kill()
        await process.wait()
