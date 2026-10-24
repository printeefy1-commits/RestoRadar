"""Private, bounded upload storage. Client names never become filesystem paths."""

import json
import os
from pathlib import Path
import re
import tempfile
import unicodedata

from fastapi import HTTPException

MAX_UPLOAD_BYTES = 100 * 1024**2
MAX_STORAGE_BYTES = 500 * 1024**2


class MediaStore:
    def __init__(self, root):
        self.root = Path(root)

    def record(self, tenant):
        folder = self.root / str(tenant)
        try:
            info = json.loads((folder / "current.json").read_text())
            if not re.fullmatch(r"[a-f0-9]{64}", info["id"]):
                return None
            path = folder / f"{info['id']}.mp4"
            if not path.is_file() or path.stat().st_size != info["bytes"]:
                return None
            return path, info
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def begin(self):
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        used = sum(p.stat().st_size for p in self.root.glob("*/*.mp4"))
        handle = tempfile.NamedTemporaryFile(
            prefix="incoming-", suffix=".part", dir=self.root, delete=False
        )
        return handle, used

    def commit(self, tenant, path, info, name):
        folder = self.root / str(tenant)
        folder.mkdir(exist_ok=True, mode=0o700)
        previous = self.record(tenant)
        target = folder / f"{info['id']}.mp4"
        # Keep only a short display name; neither it nor a client path is opened.
        name = name.replace("\\", "/").rsplit("/", 1)[-1]
        name = "".join(c for c in name if not unicodedata.category(c).startswith("C"))
        public = {**info, "uploaded": True, "name": name[:120] or "video.mp4"}
        metadata = folder / "next.json"
        try:
            metadata.write_text(json.dumps(public))
            os.replace(path, target)
            os.replace(metadata, folder / "current.json")
        except OSError:
            metadata.unlink(missing_ok=True)
            if not previous or previous[0] != target:
                target.unlink(missing_ok=True)
            raise
        if previous and previous[0] != target:
            previous[0].unlink(missing_ok=True)
        return public

    def delete(self, tenant, source_id):
        previous = self.record(tenant)
        if not previous:
            raise HTTPException(404, "لا يوجد فيديو مرفوع لحذفه")
        if previous[1]["id"] != source_id:
            raise HTTPException(409, "تغير الفيديو؛ حدّث الصفحة قبل الحذف")
        (previous[0].parent / "current.json").unlink()
        previous[0].unlink(missing_ok=True)

    def recover(self):
        # Only our staging files and superseded uploads, never recordings/models.
        if not self.root.is_dir():
            return
        for path in self.root.glob("incoming-*.part"):
            path.unlink(missing_ok=True)
        for folder in self.root.iterdir():
            if not folder.is_dir():
                continue
            record = self.record(folder.name)
            for path in folder.glob("*.mp4"):
                if not record or path != record[0]:
                    path.unlink(missing_ok=True)
            (folder / "next.json").unlink(missing_ok=True)
