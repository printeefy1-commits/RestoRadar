"""Internal-only video trial worker. Public access goes through the authenticated edge API."""

from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from uuid import UUID, uuid4
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from models import digest, prepare
from decisions import MAX_FOCUS_TABLES

ROOT = Path(__file__).resolve().parent
SOURCE = Path(os.environ.get("VIDEO_SAMPLE_PATH", "/recordings/restaurant-demo.mp4"))
lock = threading.Lock()
model_state = {"models": "DOWNLOADING"}
jobs = {}
active = None
sample_cache = None
stopping = False


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
    poseModel: Literal["RTMPOSE_TINY", "RTMPOSE_MEDIUM", "YOLO26_NANO_POSE"] = (
        "RTMPOSE_TINY"
    )
    wristTrial: bool = Field(default=False, strict=True)
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


def sample():
    global sample_cache
    if not SOURCE.is_file():
        return None
    stat = SOURCE.stat()
    signature = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
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
                "-show_entries",
                "format=duration:stream=width,height",
                "-of",
                "json",
                str(SOURCE),
            ],
            capture_output=True,
            check=True,
            timeout=15,
        )
        info = json.loads(result.stdout)
        stream = info["streams"][0]
        value = {
            "id": digest(SOURCE),
            "duration": float(info["format"]["duration"]),
            "width": int(stream["width"]),
            "height": int(stream["height"]),
            "bytes": stat.st_size,
        }
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
        if sample()["id"] != source_id:
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
    source = sample()
    if not source:
        raise HTTPException(404, "لم يوجد فيديو التجربة")
    if source["id"] != trial.sourceId:
        raise HTTPException(409, "تغير الفيديو؛ أعد فتح الصفحة وتحديد المقاعد")
    if trial.startSeconds + 5 > source["duration"]:
        raise HTTPException(422, "اختر بداية تتيح خمس ثوانٍ على الأقل من الفيديو")
    with lock:
        cleanup()
        if stopping or model_state["models"] != "READY":
            raise HTTPException(503, "نماذج التحليل ليست جاهزة بعد")
        if active or len(jobs) >= 32:
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
                        "VIDEO_SAMPLE_PATH": str(SOURCE),
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
