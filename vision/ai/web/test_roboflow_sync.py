"""Regression tests for strict Roboflow annotation synchronization."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from vision.ai.web.roboflow_sync import (
    annotation_digest,
    build_roboflow_provenance,
    canonical_local_annotation,
    canonical_remote_annotation,
    normalize_roboflow_filename,
    roboflow_image_details,
    roboflow_upload_annotation,
    redact_secret,
    sync_preview_digest,
)


class FakeResponse:
    ok = True
    status_code = 200
    text = ""

    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload


class RoboflowSyncTests(unittest.TestCase):
    def test_export_filename_normalization_is_conservative(self):
        self.assertEqual(normalize_roboflow_filename("Leaf_jpg.rf.a1b2c3.jpg"), "leaf")
        self.assertEqual(normalize_roboflow_filename("Leaf Image.JPG"), "leaf-image")

    def test_local_and_remote_boxes_have_the_same_canonical_form(self):
        with TemporaryDirectory() as directory:
            label = Path(directory) / "leaf.txt"
            label.write_text("0 0.5 0.4 0.2 0.1\n", encoding="utf-8")
            local = canonical_local_annotation(label, ["leaf"])
        remote = canonical_remote_annotation({
            "image": {
                "annotation": {
                    "width": 1000,
                    "height": 500,
                    "boxes": [{"label": "leaf", "x": 500, "y": 200, "width": 200, "height": 50}],
                }
            }
        }, ["leaf"])
        self.assertEqual(local, remote)
        self.assertEqual(annotation_digest(local), annotation_digest(remote))

    def test_manifest_rejects_multiple_local_derivatives_for_one_source_id(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images" / "train"
            labels = root / "labels" / "train"
            images.mkdir(parents=True)
            labels.mkdir(parents=True)
            for name in ("leaf_jpg.rf.first.jpg", "leaf_jpg.rf.second.jpg"):
                (images / name).write_bytes(b"image")
                (labels / f"{Path(name).stem}.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
            manifest = build_roboflow_provenance(
                root,
                {"train": "images/train"},
                ["leaf"],
                "farm",
                "leaves",
                "7",
                [{"id": "source-1", "name": "leaf.jpg"}],
            )
        self.assertEqual(manifest["workspace"], "farm")
        self.assertEqual(manifest["project"], "leaves")
        self.assertEqual(manifest["mapping"]["eligible"], 0)
        self.assertTrue(all("augmented derivative" in item["reason"] for item in manifest["images"].values()))

    def test_manifest_binds_only_a_unique_source_image(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "images" / "val").mkdir(parents=True)
            (root / "labels" / "val").mkdir(parents=True)
            (root / "images" / "val" / "leaf.jpg").write_bytes(b"image")
            (root / "labels" / "val" / "leaf.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
            manifest = build_roboflow_provenance(
                root,
                {"val": "images/val"},
                ["leaf"],
                "farm",
                "leaves",
                "7",
                [{"id": "source-1", "name": "leaf.jpg"}],
            )
        entry = manifest["images"]["images/val/leaf.jpg"]
        self.assertTrue(entry["eligible"])
        self.assertEqual(entry["roboflow_image_id"], "source-1")

    def test_exact_source_stays_eligible_while_generated_copy_is_blocked(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "images" / "train").mkdir(parents=True)
            (root / "labels" / "train").mkdir(parents=True)
            for name in ("leaf.jpg", "leaf_jpg.rf.generated.jpg"):
                (root / "images" / "train" / name).write_bytes(b"image")
                (root / "labels" / "train" / f"{Path(name).stem}.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
            manifest = build_roboflow_provenance(
                root, {"train": "images/train"}, ["leaf"], "farm", "leaves", "7",
                [{"id": "source-1", "name": "leaf.jpg"}],
            )
        exact = manifest["images"]["images/train/leaf.jpg"]
        generated = manifest["images"]["images/train/leaf_jpg.rf.generated.jpg"]
        self.assertTrue(exact["eligible"])
        self.assertFalse(generated["eligible"])
        self.assertIn("generated derivative", generated["reason"])

    def test_image_lookup_uses_bound_workspace_project_and_id(self):
        captured = {}

        def fake_get(url, **kwargs):
            captured.update(url=url, kwargs=kwargs)
            return FakeResponse({"image": {"id": "image-1"}})

        result = roboflow_image_details("secret", "my workspace", "project/one", "image-1", fake_get)
        self.assertEqual(result["image"]["id"], "image-1")
        self.assertIn("my%20workspace/project%2Fone/images/image-1", captured["url"])
        self.assertEqual(captured["kwargs"]["params"], {"api_key": "secret"})

    def test_upload_requests_explicit_overwrite_with_full_labelmap(self):
        captured = {}

        def fake_post(url, **kwargs):
            captured.update(url=url, kwargs=kwargs)
            return FakeResponse({"success": True})

        roboflow_upload_annotation(
            "secret", "leaves", "image-1", "1 0.5 0.5 0.2 0.2\n", ["leaf", "diseased"], "leaf.txt", fake_post,
        )
        self.assertEqual(captured["kwargs"]["params"]["overwrite"], "true")
        self.assertEqual(captured["kwargs"]["json"]["labelmap"], {"0": "leaf", "1": "diseased"})
        self.assertIn("/dataset/leaves/annotate/image-1", captured["url"])

    def test_preview_digest_changes_when_target_changes(self):
        items = [{"local_image": "images/val/leaf.jpg", "status": "ready", "corrected_digest": "new"}]
        first = sync_preview_digest({"workspace": "one", "project": "leaves"}, items)
        second = sync_preview_digest({"workspace": "two", "project": "leaves"}, items)
        self.assertNotEqual(first, second)

    def test_api_keys_are_redacted_from_persistable_errors(self):
        self.assertNotIn("private-key", redact_secret("request api_key=private-key failed", "private-key"))


if __name__ == "__main__":
    unittest.main()
