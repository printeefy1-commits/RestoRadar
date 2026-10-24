"""Real MP4 probing, private upload lifecycle and rollback; no model downloads."""

import asyncio
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient
import service
from media import MediaStore


class MediaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = tempfile.TemporaryDirectory()
        path = Path(cls.fixture.name) / "video.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=320x240:rate=10",
                "-t",
                "8",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-threads",
                "1",
                str(path),
            ],
            check=True,
        )
        cls.video = path.read_bytes()
        alt = path.with_name("alternative.mp4")
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(path),
                "-c",
                "copy",
                "-metadata",
                "title=alternative",
                str(alt),
            ],
            check=True,
        )
        cls.other_video = alt.read_bytes()
        camera = path.with_name("camera-hevc.mp4")
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(path),
                "-an",
                "-c:v",
                "libx265",
                "-preset",
                "ultrafast",
                "-x265-params",
                "pools=none:frame-threads=1:log-level=error",
                "-pix_fmt",
                "yuv420p",
                "-threads",
                "1",
                str(camera),
            ],
            check=True,
        )
        cls.hevc = camera.read_bytes()

    @classmethod
    def tearDownClass(cls):
        cls.fixture.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.original = self.root / "original.mp4"
        self.original.write_bytes(self.video)
        self.store = MediaStore(self.root / "uploads")
        self.tenant = str(uuid4())
        self.base = f"/restaurants/{self.tenant}"
        self.patches = [
            patch.object(service, "uploads", self.store),
            patch.object(service, "SOURCE", self.original),
            patch.object(service, "sample_cache", None),
            patch.object(service, "prepare_models"),
            patch.object(service, "media_busy", False),
            patch.object(service, "active", None),
        ]
        for p in self.patches:
            p.start()
        self.client = TestClient(service.app).__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        for p in reversed(self.patches):
            p.stop()
        self.temp.cleanup()

    def upload(self, content=None, headers=None):
        return self.client.post(
            self.base + "/video",
            content=self.video if content is None else content,
            headers={"Content-Type": "video/mp4", **(headers or {})},
        )

    def test_upload_preview_range_and_persistence_without_touching_original(self):
        response = self.upload(headers={"X-Video-Name": "..%2F..%2Fcamera.mp4"})
        self.assertEqual(response.status_code, 201, response.text)
        info = response.json()
        self.assertEqual(info["name"], "camera.mp4")
        self.assertTrue(info["uploaded"])
        self.assertEqual(info["bytes"], len(self.video))
        record = MediaStore(self.store.root).record(self.tenant)
        self.assertEqual(record[1], info)
        response = self.client.get(
            self.base + "/video?source=" + info["id"], headers={"Range": "bytes=0-10"}
        )
        self.assertEqual(response.status_code, 206)
        self.assertEqual(response.content, self.video[:11])
        self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertEqual(self.client.get(self.base + "/status").json()["source"], info)
        self.assertEqual(self.original.read_bytes(), self.video)
        self.assertFalse(service.media_busy)

    def test_other_tenant_cannot_read_delete_or_select_upload(self):
        info = self.upload(self.other_video).json()
        other = f"/restaurants/{uuid4()}"
        self.assertEqual(
            self.client.get(other + "/video?source=" + info["id"]).status_code, 409
        )
        self.assertEqual(
            self.client.post(
                other + "/video/delete", json={"sourceId": info["id"]}
            ).status_code,
            404,
        )
        self.assertNotIn(
            "uploaded", self.client.get(other + "/status").json()["source"]
        )
        self.assertIsNotNone(self.store.record(self.tenant))

    def test_delete_is_source_checked_and_restores_only_original(self):
        info = self.upload(self.other_video).json()
        self.assertEqual(
            self.client.post(
                self.base + "/video/delete", json={"sourceId": "0" * 64}
            ).status_code,
            409,
        )
        self.assertIsNotNone(self.store.record(self.tenant))
        response = self.client.post(
            self.base + "/video/delete", json={"sourceId": info["id"]}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(list(self.store.root.glob("*/*.mp4")), [])
        self.assertEqual(
            self.client.get(self.base + "/status").json()["source"]["id"],
            service.digest(self.original),
        )
        self.assertEqual(self.original.read_bytes(), self.video)
        self.assertEqual(
            self.client.post(
                self.base + "/video/delete", json={"sourceId": info["id"]}
            ).status_code,
            404,
        )

    def test_valid_replacement_removes_only_superseded_upload(self):
        old = self.upload().json()
        new = self.upload(self.other_video).json()
        self.assertNotEqual(old["id"], new["id"])
        self.assertEqual(self.store.record(self.tenant)[1], new)
        self.assertEqual(len(list(self.store.root.glob("*/*.mp4"))), 1)
        self.assertEqual(
            self.client.get(self.base + "/video?source=" + old["id"]).status_code, 409
        )
        self.assertEqual(self.original.read_bytes(), self.video)

    def test_invalid_empty_truncated_and_unsupported_content_keep_current(self):
        old = self.upload().json()
        for content, headers, status in [
            (b"", {}, 422),
            (b"not a video", {}, 422),
            (self.video[:100], {}, 422),
            (self.video, {"Content-Type": "application/json"}, 415),
            (self.video, {"Content-Length": str(len(self.video) + 1)}, 422),
        ]:
            with self.subTest(status=status, length=len(content)):
                self.assertEqual(self.upload(content, headers).status_code, status)
                self.assertEqual(self.store.record(self.tenant)[1], old)
                self.assertFalse(service.media_busy)
                self.assertEqual(list(self.store.root.glob("*.part")), [])

    def test_header_and_chunked_limits_are_enforced_and_cleanup_staging(self):
        with patch.object(service, "MAX_UPLOAD_BYTES", 1024):
            self.assertEqual(
                self.upload(headers={"Content-Length": "2048"}).status_code, 413
            )
            self.assertEqual(
                self.upload(headers={"Content-Length": "invalid"}).status_code, 413
            )
            response = self.client.post(
                self.base + "/video",
                content=iter([b"x" * 700, b"y" * 700]),
                headers={"Content-Type": "video/mp4"},
            )
            self.assertEqual(response.status_code, 413)
        self.assertIsNone(self.store.record(self.tenant))
        self.assertEqual(list(self.store.root.glob("*.part")), [])
        self.assertFalse(service.media_busy)

    def test_disk_budget_rejects_replacement_without_losing_previous(self):
        old = self.upload().json()
        with patch.object(service, "MAX_STORAGE_BYTES", len(self.video) + 10):
            self.assertEqual(self.upload(self.other_video).status_code, 507)
        self.assertEqual(self.store.record(self.tenant)[1], old)
        self.assertEqual(list(self.store.root.glob("*.part")), [])

    def test_analysis_and_upload_reservation_block_mutations(self):
        info = self.upload().json()
        for key, value in [("active", object()), ("media_busy", True)]:
            with patch.object(service, key, value):
                self.assertEqual(self.upload().status_code, 409)
                self.assertEqual(
                    self.client.post(
                        self.base + "/video/delete", json={"sourceId": info["id"]}
                    ).status_code,
                    409,
                )
        self.assertEqual(self.store.record(self.tenant)[1], info)

    def test_disconnect_removes_staging_and_releases_reservation(self):
        old = self.upload().json()

        class Interrupted:
            headers = {"content-type": "video/mp4"}

            async def stream(self):
                yield b"partial"
                raise RuntimeError("simulated disconnect")

        with self.assertRaisesRegex(RuntimeError, "disconnect"):
            asyncio.run(service.upload_video(self.tenant, Interrupted()))
        self.assertEqual(self.store.record(self.tenant)[1], old)
        self.assertEqual(list(self.store.root.glob("*.part")), [])
        self.assertFalse(service.media_busy)

    def test_client_disconnect_returns_a_controlled_error_and_keeps_previous(self):
        from starlette.requests import ClientDisconnect
        from fastapi import HTTPException

        old = self.upload().json()

        class Interrupted:
            headers = {"content-type": "video/mp4"}

            async def stream(self):
                yield b"partial"
                raise ClientDisconnect()

        with self.assertRaises(HTTPException) as error:
            asyncio.run(service.upload_video(self.tenant, Interrupted()))
        self.assertEqual(error.exception.status_code, 400)
        self.assertEqual(self.store.record(self.tenant)[1], old)
        self.assertEqual(list(self.store.root.glob("*.part")), [])
        self.assertFalse(service.media_busy)

    def test_cancel_after_receiving_file_does_not_commit_validated_replacement(self):
        from fastapi import HTTPException

        old = self.upload().json()
        content = self.other_video

        class CancelledDuringValidation:
            headers = {"content-type": "video/mp4"}

            async def stream(self):
                yield content

            async def is_disconnected(self):
                return True

        with self.assertRaises(HTTPException) as error:
            asyncio.run(service.upload_video(self.tenant, CancelledDuringValidation()))
        self.assertEqual(error.exception.status_code, 400)
        self.assertEqual(self.store.record(self.tenant)[1], old)
        self.assertEqual(len(list(self.store.root.glob("*/*.mp4"))), 1)
        self.assertEqual(list(self.store.root.glob("*.part")), [])
        self.assertFalse(service.media_busy)

    def test_failed_atomic_metadata_commit_rolls_back_new_file(self):
        old = self.upload().json()
        import media

        real = media.os.replace

        def fail_metadata(source, target):
            if Path(target).name == "current.json":
                raise OSError("simulated full disk")
            return real(source, target)

        with patch.object(media.os, "replace", fail_metadata):
            self.assertEqual(self.upload(self.other_video).status_code, 507)
        self.assertEqual(self.store.record(self.tenant)[1], old)
        self.assertEqual(len(list(self.store.root.glob("*/*.mp4"))), 1)
        self.assertFalse(service.media_busy)

    def test_restart_recovers_current_and_cleans_only_orphan_uploads(self):
        info = self.upload().json()
        orphan = self.store.root / self.tenant / ("0" * 64 + ".mp4")
        orphan.write_bytes(b"orphan")
        (self.store.root / "incoming-aborted.part").write_bytes(b"partial")
        new_store = MediaStore(self.store.root)
        new_store.recover()
        self.assertEqual(new_store.record(self.tenant)[1], info)
        self.assertFalse(orphan.exists())
        self.assertEqual(list(self.store.root.glob("*.part")), [])
        self.assertEqual(self.original.read_bytes(), self.video)

    def test_upload_works_with_no_original_file(self):
        self.original.unlink()
        self.assertIsNone(self.client.get(self.base + "/status").json()["source"])
        info = self.upload().json()
        self.assertEqual(self.client.get(self.base + "/status").json()["source"], info)
        self.client.post(self.base + "/video/delete", json={"sourceId": info["id"]})
        self.assertIsNone(self.client.get(self.base + "/status").json()["source"])

    def test_unsupported_codec_is_rejected_before_commit(self):
        info = {
            "format": {"format_name": "mov,mp4,m4a,3gp,3g2,mj2", "duration": "8"},
            "streams": [
                {
                    "width": 320,
                    "height": 240,
                    "codec_name": "av1",
                    "pix_fmt": "yuv420p",
                }
            ],
        }
        with patch.object(
            service.subprocess,
            "run",
            return_value=subprocess.CompletedProcess([], 0, stdout=json.dumps(info)),
        ):
            self.assertEqual(self.upload().status_code, 422)
        self.assertIsNone(self.store.record(self.tenant))

    def test_hevc_upload_is_prepared_once_and_stored_as_browser_h264(self):
        old = self.upload().json()
        result = self.upload(self.hevc, {"X-Video-Name": "xiaomi.mp4"})
        self.assertEqual(result.status_code, 201, result.text)
        info = result.json()
        self.assertTrue(info["converted"])
        self.assertEqual(info["originalCodec"], "hevc")
        self.assertEqual(info["codec"], "h264")
        self.assertEqual(info["pixelFormat"], "yuv420p")
        self.assertAlmostEqual(info["duration"], 8, delta=0.2)
        self.assertNotEqual(info["id"], old["id"])
        path, stored = self.store.record(self.tenant)
        self.assertEqual(stored, info)
        self.assertEqual(service.sample(path, True)["codec"], "h264")
        self.assertEqual(len(list(self.store.root.glob("*/*.mp4"))), 1)
        self.assertEqual(list(self.store.root.glob("*.part")), [])
        self.assertEqual(self.original.read_bytes(), self.video)
        self.assertFalse(service.media_busy)

    def test_failed_conversion_preserves_current_video_and_cleans_both_staging_files(
        self,
    ):
        from fastapi import HTTPException

        old = self.upload().json()

        async def fail(*args):
            raise HTTPException(422, "conversion failed")

        with patch.object(service, "prepare_browser_video", fail):
            self.assertEqual(self.upload(self.hevc).status_code, 422)
        self.assertEqual(self.store.record(self.tenant)[1], old)
        self.assertEqual(list(self.store.root.glob("*.part")), [])
        self.assertFalse(service.media_busy)

    def test_cancel_during_conversion_terminates_child_and_preserves_current(self):
        from fastapi import HTTPException
        import video_conversion

        old = self.upload().json()
        content = self.hevc
        processes = []
        real = video_conversion.asyncio.create_subprocess_exec

        async def spawn(*args, **kwargs):
            process = await real(*args, **kwargs)
            processes.append(process)
            return process

        class Cancelled:
            headers = {"content-type": "video/mp4"}
            calls = 0

            async def stream(self):
                yield content

            async def is_disconnected(self):
                self.calls += 1
                return self.calls >= 2

        with patch.object(video_conversion.asyncio, "create_subprocess_exec", spawn):
            with self.assertRaises(HTTPException) as error:
                asyncio.run(service.upload_video(self.tenant, Cancelled()))
        self.assertEqual(error.exception.status_code, 400)
        self.assertEqual(len(processes), 1)
        self.assertIsNotNone(processes[0].returncode)
        self.assertEqual(self.store.record(self.tenant)[1], old)
        self.assertEqual(list(self.store.root.glob("*.part")), [])
        self.assertFalse(service.media_busy)

    def test_conversion_timeout_terminates_and_cleans_without_replacing(self):
        import video_conversion

        old = self.upload().json()
        with patch.object(video_conversion, "CONVERSION_TIMEOUT_SECONDS", 0):
            self.assertEqual(self.upload(self.hevc).status_code, 504)
        self.assertEqual(self.store.record(self.tenant)[1], old)
        self.assertEqual(list(self.store.root.glob("*.part")), [])
        self.assertFalse(service.media_busy)

    def test_conversion_requires_storage_for_both_input_and_output(self):
        old = self.upload().json()
        with patch.object(
            service, "MAX_STORAGE_BYTES", len(self.video) + len(self.hevc) + 1024
        ):
            self.assertEqual(self.upload(self.hevc).status_code, 507)
        self.assertEqual(self.store.record(self.tenant)[1], old)
        self.assertEqual(list(self.store.root.glob("*.part")), [])
        self.assertFalse(service.media_busy)

    def test_camera_clip_over_two_minutes_is_rejected_before_conversion(self):
        old = self.upload().json()
        fake = {**old, "codec": "hevc", "duration": 121}
        with patch.object(service, "sample", return_value=fake), patch.object(
            service, "prepare_browser_video"
        ) as convert:
            self.assertEqual(self.upload(self.hevc).status_code, 422)
            convert.assert_not_called()
        self.assertEqual(self.store.record(self.tenant)[1], old)
        self.assertEqual(list(self.store.root.glob("*.part")), [])


if __name__ == "__main__":
    unittest.main()
