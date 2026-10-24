"""Internal-only video trial worker. Public access goes through the authenticated edge API."""

from contextlib import asynccontextmanager
import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from urllib.parse import unquote
from uuid import UUID, uuid4
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool
from starlette.requests import ClientDisconnect
from pydantic import BaseModel, ConfigDict, Field, model_validator

from models import digest, prepare
from decisions import MAX_FOCUS_TABLES
from media import MAX_UPLOAD_BYTES, MAX_STORAGE_BYTES, MediaStore
from video_conversion import MAX_CONVERSION_SECONDS, prepare_browser_video

ROOT = Path(__file__).resolve().parent
SOURCE = Path(os.environ.get("VIDEO_SAMPLE_PATH", "/recordings/restaurant-demo.mp4"))
lock = threading.Lock()
model_state = {"models": "DOWNLOADING"}
jobs = {}
active = None
sample_cache = None
stopping = False
media_busy = False
uploads = MediaStore(os.environ.get("VIDEO_UPLOAD_DIR", "/uploads"))


class Region(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: UUID
    tableId: UUID
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    width: float = Field(ge=0.01, le=1)
    height: float = Field(ge=0.01, le=1)

    @model_validator(mode="after")
    def in_bounds(self):
        if self.x + self.width > 1.000001 or self.y + self.height > 1.000001:
            raise ValueError("Region is outside video")
        return self


class Trial(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sourceId: str = Field(pattern=r"^[a-f0-9]{64}$")
    startSeconds: float = Field(ge=0, le=86400)
    durationSeconds: int = Field(ge=6, le=120)
    regions: list[Region] = Field(min_length=1, max_length=40)
    detectionMode: Literal["FULL_FRAME", "SEAT_FOCUS"] = "FULL_FRAME"
    poseModel: Literal[
        "RTMPOSE_TINY", "RTMPOSE_MEDIUM", "YOLO26_NANO_POSE", "YOLO26_SMALL_POSE"
    ] = "RTMPOSE_TINY"
    wristTrial: bool = Field(default=False, strict=True)
    seatTrial: bool = Field(default=False, strict=True)
    tabletopRegions: list[Region] = Field(default_factory=list, max_length=40)

    @model_validator(mode="after")
    def unique_regions(self):
        all_regions = self.regions + self.tabletopRegions
        if len({r.id for r in all_regions}) != len(all_regions):
            raise ValueError("Repeated regions")
        if self.wristTrial and not self.tabletopRegions:
            raise ValueError("ارسم منطقة سطح الطاولة لتجربة المعصمين")
        if not self.wristTrial and self.tabletopRegions:
            raise ValueError("فعّل تجربة المعصمين لاستخدام مناطق سطح الطاولة")
        if any(
            r.tableId not in {s.tableId for s in self.regions}
            for r in self.tabletopRegions
        ):
            raise ValueError("اربط سطح الطاولة بمنطقة مقعد للطاولة نفسها")
        if (
            self.detectionMode == "SEAT_FOCUS"
            and len({r.tableId for r in self.regions}) > MAX_FOCUS_TABLES
        ):
            raise ValueError(
                "الرصد المركّز يدعم أربع طاولات كحد أقصى في التجربة الواحدة"
            )
        return self


def sample(path=None, upload=False):
    global sample_cache
    path = SOURCE if path is None else Path(path)
    if not path.is_file():
        return None
    stat = path.stat()
    signature = (
        str(path),
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
        upload,
    )
    if sample_cache and sample_cache[0] == signature:
        return sample_cache[1]
    if stat.st_size > 2 * 1024**3:
        raise HTTPException(422, "حجم الفيديو يتجاوز حد التجربة البالغ 2 GiB")
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-protocol_whitelist",
                "file",
                "-show_entries",
                "format=duration,format_name:stream=width,height,codec_name,pix_fmt",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            check=True,
            timeout=15,
        )
        info = json.loads(result.stdout)
        stream = info["streams"][0]
        if upload and (
            "mp4" not in info["format"]["format_name"].split(",")
            or stream["codec_name"] not in ("h264", "hevc")
        ):
            raise HTTPException(422, "استخدم فيديو MP4 بترميز H.264 أو H.265/HEVC")
        value = {
            "id": digest(path),
            "duration": float(info["format"]["duration"]),
            "width": int(stream["width"]),
            "height": int(stream["height"]),
            "bytes": stat.st_size,
        }
        if upload:
            value.update(codec=stream["codec_name"], pixelFormat=stream["pix_fmt"])
        if (
            not 0 < value["duration"] <= 86400
            or min(value["width"], value["height"]) < 100
            or max(value["width"], value["height"]) > 4096
        ):
            raise ValueError("Unsupported video dimensions/duration")
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, IndexError):
        raise HTTPException(
            422, "تعذر قراءة الفيديو؛ استخدم MP4 صالحًا بدقة لا تتجاوز 4K"
        ) from None
    sample_cache = (signature, value)
    return value


def selected_source(tenant):
    record = uploads.record(tenant)
    return record if record else (SOURCE, sample())


class DeleteVideo(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sourceId: str = Field(pattern=r"^[a-f0-9]{64}$")


def prepare_models():
    with lock:
        model_state.clear()
        model_state["models"] = "DOWNLOADING"
    try:
        prepare()
        with lock:
            model_state["models"] = "READY"
    except Exception as error:
        print(f"Model preparation failed: {type(error).__name__}: {error}", flush=True)
        # Do not return remote response bodies, filesystem paths or environment details.
        with lock:
            model_state.update(
                models="ERROR",
                message="تعذر تنزيل نماذج التحليل. تحقق من اتصال الخادم ثم أعد المحاولة.",
            )


def cleanup():
    for job_id, job in list(jobs.items()):
        if job["status"] != "RUNNING" and time.monotonic() - job["created"] > 3600:
            job["temp"].cleanup()
            del jobs[job_id]


def public_job(job):
    result = {k: job[k] for k in ("id", "status", "progress")}
    if job.get("message"):
        result["message"] = job["message"]
    if job["status"] == "RUNNING":
        try:
            progress = json.loads((job["path"] / "progress.json").read_text())
            result["progress"] = progress["progress"]
            if progress.get("phase") == "PREPARING_POSE":
                result["message"] = (
                    "جارٍ تجهيز نموذج تقدير الجلوس؛ قد يتضمن تنزيله لأول مرة."
                )
        except (OSError, ValueError, KeyError, TypeError):
            pass
    if job["status"] == "COMPLETED":
        result["report"] = json.loads((job["path"] / "result.json").read_text())
    return result


def monitor(job, source_id):
    global active
    try:
        returncode = job["process"].wait(timeout=600)
        if returncode != 0 or not (job["path"] / "result.json").is_file():
            raise ValueError("Analysis process failed")
        if sample(job.get("sourcePath", SOURCE))["id"] != source_id:
            raise ValueError("Source changed during analysis")
        with lock:
            job.update(status="COMPLETED", progress=1)
    except Exception as error:
        job["process"].kill()
        job["process"].wait()
        log = job["path"] / "error.log"
        detail = log.read_text(errors="replace")[-3000:] if log.is_file() else ""
        print(
            f"Video trial {job['id']} failed ({type(error).__name__}): {detail}",
            flush=True,
        )
        message = "لم يكتمل التحليل. تحقق من صلاحية الفيديو والنماذج أو جرّب مدة أقصر."
        try:
            progress = json.loads((job["path"] / "progress.json").read_text())
            if progress.get("phase") == "PREPARING_POSE":
                message = (
                    "تعذر تجهيز نموذج تقدير الجلوس المختار. "
                    "يمكنك اختيار النموذج الخفيف الحالي، أو إعادة محاولة النموذج الأكبر."
                )
        except (OSError, ValueError, TypeError):
            pass
        with lock:
            job.update(
                status="FAILED",
                message=message,
            )
    finally:
        with lock:
            active = None


@asynccontextmanager
async def lifespan(_app):
    global stopping
    stopping = False
    uploads.recover()
    thread = threading.Thread(target=prepare_models, daemon=True)
    thread.start()
    yield
    with lock:
        stopping = True
        if active and active["process"].poll() is None:
            active["process"].kill()
            active["process"].wait()


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/status")
def status():
    with lock:
        models = dict(model_state)
    return {"enabled": True, **models, "source": sample()}


@app.get("/restaurants/{restaurant_id}/status")
def restaurant_status(restaurant_id: UUID):
    with lock:
        return {
            "enabled": True,
            **model_state,
            "source": selected_source(restaurant_id)[1],
            "maxUploadBytes": MAX_UPLOAD_BYTES,
            "mediaBusy": media_busy,
        }


@app.get("/restaurants/{restaurant_id}/video")
def restaurant_video(restaurant_id: UUID, source: str | None = None):
    with lock:
        path, info = selected_source(restaurant_id)
        if not info:
            raise HTTPException(404, "لم يوجد فيديو التجربة")
        if source is not None and source != info["id"]:
            raise HTTPException(409, "تغير الفيديو؛ حدّث الصفحة")
        return FileResponse(
            path, media_type="video/mp4", headers={"Cache-Control": "no-store"}
        )


@app.post("/restaurants/{restaurant_id}/video", status_code=201)
async def upload_video(restaurant_id: UUID, request: Request):
    global media_busy
    if request.headers.get("content-type") != "video/mp4":
        raise HTTPException(415, "استخدم فيديو MP4")
    length = request.headers.get("content-length")
    if length and (not length.isdigit() or int(length) > MAX_UPLOAD_BYTES):
        raise HTTPException(413, "الحد الأقصى للفيديو 100 ميغابايت")
    with lock:
        if active or media_busy or stopping:
            raise HTTPException(409, "انتظر اكتمال التحليل أو رفع الفيديو الجاري")
        media_busy = True
    path = converted = None
    try:
        handle, used = uploads.begin()
        path = Path(handle.name)
        size = 0
        async with asyncio.timeout(120):
            with handle:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > MAX_UPLOAD_BYTES:
                        raise HTTPException(413, "الحد الأقصى للفيديو 100 ميغابايت")
                    if used + size > MAX_STORAGE_BYTES:
                        raise HTTPException(
                            507, "مساحة الفيديوهات المرفوعة ممتلئة؛ احذف فيديو أولًا"
                        )
                    handle.write(chunk)
            if not size or (length and size != int(length)):
                raise HTTPException(422, "لم يصل ملف فيديو كامل")
        info = await run_in_threadpool(sample, path, True)
        if info["duration"] < 6:
            raise HTTPException(422, "اختر فيديو مدته ست ثوانٍ على الأقل")
        if await request.is_disconnected():
            raise HTTPException(400, "أُلغي الرفع قبل اعتماد الفيديو")
        ready_path = path
        if info["codec"] != "h264" or info["pixelFormat"] not in (
            "yuv420p",
            "yuvj420p",
        ):
            if info["duration"] > MAX_CONVERSION_SECONDS:
                raise HTTPException(
                    422,
                    "التهيئة التلقائية تدعم مقطعًا حتى 120 ثانية؛ قصّر المقطع ثم ارفعه",
                )
            storage_limit = MAX_STORAGE_BYTES - used - size
            if storage_limit <= 64 * 1024:
                raise HTTPException(
                    507, "لا توجد مساحة كافية لتهيئة الفيديو؛ احذف فيديو أولًا"
                )
            output_limit = min(MAX_UPLOAD_BYTES, storage_limit)
            handle, _ = uploads.begin()
            converted = Path(handle.name)
            handle.close()
            await prepare_browser_video(path, converted, output_limit, request)
            output_info = await run_in_threadpool(sample, converted, True)
            if (
                output_info["bytes"] >= output_limit - 64 * 1024
                or abs(output_info["duration"] - info["duration"]) > 0.5
            ):
                raise HTTPException(
                    413, "النسخة المجهزة تجاوزت المساحة المتاحة؛ جرّب مقطعًا أقصر"
                )
            if output_info["codec"] != "h264" or output_info["pixelFormat"] not in (
                "yuv420p",
                "yuvj420p",
            ):
                raise HTTPException(422, "لم تكتمل تهيئة الفيديو بصيغة متوافقة")
            info = {**output_info, "converted": True, "originalCodec": info["codec"]}
            ready_path = converted
        if await request.is_disconnected():
            raise HTTPException(400, "أُلغي الرفع قبل اعتماد الفيديو")
        with lock:
            return uploads.commit(
                restaurant_id,
                ready_path,
                info,
                unquote(request.headers.get("x-video-name", "video.mp4")),
            )
    except ClientDisconnect:
        raise HTTPException(400, "انقطع الاتصال قبل اكتمال رفع الفيديو") from None
    except TimeoutError:
        raise HTTPException(
            408, "انتهت مهلة الرفع؛ جرّب ملفًا أصغر أو اتصالًا أسرع"
        ) from None
    except OSError:
        raise HTTPException(507, "تعذر حفظ الفيديو؛ تحقق من المساحة المتاحة") from None
    finally:
        try:
            try:
                if converted:
                    converted.unlink(missing_ok=True)
            finally:
                if path:
                    path.unlink(missing_ok=True)
        finally:
            with lock:
                media_busy = False


@app.post("/restaurants/{restaurant_id}/video/delete")
def delete_video(restaurant_id: UUID, body: DeleteVideo):
    with lock:
        if active or media_busy:
            raise HTTPException(409, "انتظر اكتمال التحليل أو رفع الفيديو الجاري")
        uploads.delete(restaurant_id, body.sourceId)
        return {"ok": True}


@app.post("/models/retry")
def retry():
    with lock:
        if model_state["models"] != "ERROR":
            raise HTTPException(409, "تهيئة النماذج جارية أو مكتملة")
        model_state["models"] = "DOWNLOADING"
    threading.Thread(target=prepare_models, daemon=True).start()
    return {"ok": True}


@app.get("/video")
def video():
    if not sample():
        raise HTTPException(404, "لم يوجد فيديو التجربة على الخادم")
    return FileResponse(
        SOURCE, media_type="video/mp4", headers={"Cache-Control": "no-store"}
    )


@app.post("/restaurants/{restaurant_id}/jobs", status_code=202)
def start(restaurant_id: UUID, trial: Trial):
    global active
    with lock:
        cleanup()
        source_path, source = selected_source(restaurant_id)
        if not source:
            raise HTTPException(404, "لم يوجد فيديو التجربة")
        if source["id"] != trial.sourceId:
            raise HTTPException(409, "تغير الفيديو؛ أعد فتح الصفحة وتحديد المقاعد")
        if trial.startSeconds + 5 > source["duration"]:
            raise HTTPException(422, "اختر بداية تتيح خمس ثوانٍ على الأقل من الفيديو")
        if stopping or model_state["models"] != "READY":
            raise HTTPException(503, "نماذج التحليل ليست جاهزة بعد")
        if active or media_busy or len(jobs) >= 32:
            raise HTTPException(409, "يوجد تحليل جارٍ؛ انتظر اكتماله")
        job_id = str(uuid4())
        directory = tempfile.TemporaryDirectory(prefix="restoradar-trial-")
        path = Path(directory.name)
        (path / "config.json").write_text(trial.model_dump_json())
        try:
            with (path / "error.log").open("wb") as log:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        str(ROOT / "analyze.py"),
                        str(path / "config.json"),
                        str(path / "result.json"),
                        str(path / "progress.json"),
                    ],
                    cwd=ROOT,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=log,
                    env={
                        **os.environ,
                        "VIDEO_SAMPLE_PATH": str(source_path),
                        "OMP_NUM_THREADS": "2",
                    },
                )
        except OSError:
            directory.cleanup()
            raise HTTPException(503, "تعذر بدء عملية التحليل") from None
        job = {
            "id": job_id,
            "tenant": str(restaurant_id),
            "status": "RUNNING",
            "progress": 0,
            "created": time.monotonic(),
            "path": path,
            "temp": directory,
            "process": process,
            "sourcePath": source_path,
        }
        jobs[job_id] = job
        active = job
        threading.Thread(target=monitor, args=(job, source["id"]), daemon=True).start()
        return public_job(job)


@app.get("/restaurants/{restaurant_id}/jobs/latest")
def latest(restaurant_id: UUID):
    with lock:
        cleanup()
        selected = [j for j in jobs.values() if j["tenant"] == str(restaurant_id)]
        return (
            public_job(max(selected, key=lambda j: j["created"])) if selected else None
        )


@app.get("/restaurants/{restaurant_id}/jobs/{job_id}")
def get_job(restaurant_id: UUID, job_id: UUID):
    with lock:
        cleanup()
        job = jobs.get(str(job_id))
        if not job or job["tenant"] != str(restaurant_id):
            raise HTTPException(
                404, "لم يوجد هذا التقرير أو انتهت مدة الاحتفاظ المؤقتة"
            )
        return public_job(job)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, access_log=False)
