"""Tests for deterministic YOLO multi-label stratification."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from .stratified_split import (
    largest_remainder_counts,
    read_yolo_profile,
    stratified_split,
)


class StratifiedSplitTests(unittest.TestCase):
    def build_items(self, root: Path, label_sets: list[list[int]]):
        image_dir = root / "images"
        label_dir = root / "labels"
        image_dir.mkdir()
        label_dir.mkdir()
        items = []
        for index, class_ids in enumerate(label_sets):
            image_path = image_dir / f"image_{index}.jpg"
            image_path.touch()
            rows = [f"{class_id} 0.5 0.5 0.1 0.1" for class_id in class_ids]
            (label_dir / f"image_{index}.txt").write_text(
                "\n".join(rows),
                encoding="utf-8",
            )
            items.append((image_path, label_dir))
        return items

    def test_largest_remainder_produces_exact_capacities(self):
        ratios = {"train": 0.7, "val": 0.15, "test": 0.15}
        self.assertEqual(
            largest_remainder_counts(911, ratios),
            {"train": 638, "val": 137, "test": 136},
        )

    def test_split_is_exact_and_deterministic(self):
        ratios = {"train": 0.7, "val": 0.15, "test": 0.15}
        with TemporaryDirectory() as directory:
            root = Path(directory)
            items = self.build_items(
                root,
                [
                    [0, 1, 2] if index < 3
                    else [0, 1] if index < 12
                    else [0] if index < 27
                    else []
                    for index in range(30)
                ],
            )
            first, _ = stratified_split(items, ratios, {0, 1, 2}, seed=42)
            second, _ = stratified_split(items, ratios, {0, 1, 2}, seed=42)

            self.assertEqual(
                {name: len(group) for name, group in first.items()},
                {"train": 21, "val": 5, "test": 4},
            )
            self.assertEqual(first, second)
            assigned = [item for group in first.values() for item in group]
            self.assertEqual(len(assigned), len(items))
            self.assertEqual(len({item[0] for item in assigned}), len(items))

    def test_rare_class_is_represented_when_feasible(self):
        ratios = {"train": 0.7, "val": 0.15, "test": 0.15}
        with TemporaryDirectory() as directory:
            root = Path(directory)
            items = self.build_items(
                root,
                [[0, 1] if index < 3 else [0] for index in range(20)],
            )
            groups, _ = stratified_split(items, ratios, {0, 1}, seed=42)
            rare_counts = {}
            for split, pairs in groups.items():
                rare_counts[split] = sum(
                    1 in read_yolo_profile(
                        label_dir / f"{image_path.stem}.txt",
                        {0, 1},
                    )[0]
                    for image_path, label_dir in pairs
                )
            self.assertEqual(rare_counts, {"train": 1, "val": 1, "test": 1})

    def test_progress_reports_each_stratification_stage(self):
        ratios = {"train": 0.7, "val": 0.15, "test": 0.15}
        with TemporaryDirectory() as directory:
            items = self.build_items(Path(directory), [[0], [0, 1], [], [1]])
            updates = []
            stratified_split(
                items,
                ratios,
                {0, 1},
                progress_callback=lambda stage, current, total: updates.append((stage, current, total)),
            )

            reading = [(current, total) for stage, current, total in updates if stage == "reading_labels"]
            assigning = [(current, total) for stage, current, total in updates if stage == "assigning"]
            self.assertEqual(reading, [(current, len(items)) for current in range(1, len(items) + 1)])
            self.assertEqual(assigning, [(current, len(items)) for current in range(1, len(items) + 1)])
            self.assertIn(("calculating_targets", 1, 1), updates)
            self.assertEqual(updates[-1], ("finalizing_split", 1, 1))

    def test_common_and_rare_class_proportions_follow_requested_ratios(self):
        ratios = {"train": 0.7, "val": 0.15, "test": 0.15}
        with TemporaryDirectory() as directory:
            root = Path(directory)
            items = self.build_items(
                root,
                [[0, 1] if index < 20 else [0] for index in range(100)],
            )
            groups, _ = stratified_split(items, ratios, {0, 1}, seed=42)

            class_counts = {name: {0: 0, 1: 0} for name in groups}
            for split, pairs in groups.items():
                for image_path, label_dir in pairs:
                    labels, _ = read_yolo_profile(
                        label_dir / f"{image_path.stem}.txt",
                        {0, 1},
                    )
                    for class_id in labels:
                        class_counts[split][class_id] += 1

            self.assertEqual(class_counts["train"], {0: 70, 1: 14})
            self.assertEqual(class_counts["val"], {0: 15, 1: 3})
            self.assertEqual(class_counts["test"], {0: 15, 1: 3})

    def test_instance_heavy_images_are_balanced(self):
        ratios = {"train": 0.5, "val": 0.5, "test": 0.0}
        with TemporaryDirectory() as directory:
            root = Path(directory)
            items = self.build_items(
                root,
                [[0] * count for count in (10, 10, 1, 1, 1, 1)],
            )
            groups, _ = stratified_split(items, ratios, {0}, seed=42)
            instance_counts = {}
            for split, pairs in groups.items():
                instance_counts[split] = sum(
                    read_yolo_profile(
                        label_dir / f"{image_path.stem}.txt",
                        {0},
                    )[1][0]
                    for image_path, label_dir in pairs
                )

            self.assertEqual(instance_counts, {"train": 12, "val": 12, "test": 0})


if __name__ == "__main__":
    unittest.main()
