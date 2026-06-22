"""Deterministic multi-label stratification for YOLO detection datasets."""

from collections import Counter
import math
from pathlib import Path
import random


SPLIT_NAMES = ("train", "val", "test")


def largest_remainder_counts(total: int, ratios: dict[str, float]) -> dict[str, int]:
    """Convert fractional split ratios into exact integer capacities."""
    raw = {name: total * ratios[name] for name in SPLIT_NAMES}
    counts = {name: math.floor(raw[name]) for name in SPLIT_NAMES}
    remaining = total - sum(counts.values())
    order = sorted(
        SPLIT_NAMES,
        key=lambda name: (raw[name] - counts[name], -SPLIT_NAMES.index(name)),
        reverse=True,
    )
    for name in order[:remaining]:
        counts[name] += 1
    return counts


def read_yolo_profile(label_path: Path, valid_class_ids: set[int]) -> tuple[set[int], Counter]:
    """Return image-level class presence and per-class instance counts."""
    instances = Counter()
    if not label_path.is_file():
        return set(), instances

    for line in label_path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.strip().split()
        if not fields:
            continue
        try:
            class_id = int(fields[0])
        except ValueError:
            continue
        if class_id in valid_class_ids:
            instances[class_id] += 1
    return set(instances), instances


def class_presence_targets(
    occurrences: int,
    ratios: dict[str, float],
    capacities: dict[str, int],
) -> dict[str, int]:
    """Allocate per-class image targets, prioritizing rare-class coverage."""
    targets = largest_remainder_counts(occurrences, ratios)
    eligible = [name for name in SPLIT_NAMES if ratios[name] > 0 and capacities[name] > 0]
    required = eligible[:min(occurrences, len(eligible))]
    raw = {name: occurrences * ratios[name] for name in SPLIT_NAMES}

    for recipient in required:
        if targets[recipient] > 0:
            continue
        donors = [
            name for name in SPLIT_NAMES
            if targets[name] > (1 if name in required else 0)
        ]
        if not donors:
            continue
        donor = max(
            donors,
            key=lambda name: (
                targets[name] - raw[name],
                targets[name],
                -SPLIT_NAMES.index(name),
            ),
        )
        targets[donor] -= 1
        targets[recipient] += 1
    return targets


def stratified_split(
    items: list[tuple[Path, Path]],
    ratios: dict[str, float],
    valid_class_ids: set[int],
    seed: int = 42,
) -> tuple[dict[str, list[tuple[Path, Path]]], dict]:
    """Split YOLO images while preserving multi-label class proportions."""
    capacities = largest_remainder_counts(len(items), ratios)
    records = []
    for index, (image_path, label_dir) in enumerate(items):
        labels, instances = read_yolo_profile(
            label_dir / f"{image_path.stem}.txt",
            valid_class_ids,
        )
        records.append({
            "index": index,
            "item": (image_path, label_dir),
            "labels": labels,
            "instances": instances,
        })

    presence_totals = Counter()
    instance_totals = Counter()
    members: dict[int, set[int]] = {class_id: set() for class_id in valid_class_ids}
    for record in records:
        for class_id in record["labels"]:
            presence_totals[class_id] += 1
            members[class_id].add(record["index"])
        instance_totals.update(record["instances"])

    presence_targets = {
        class_id: class_presence_targets(presence_totals[class_id], ratios, capacities)
        for class_id in valid_class_ids
    }
    instance_targets = {
        class_id: {
            name: instance_totals[class_id] * ratios[name]
            for name in SPLIT_NAMES
        }
        for class_id in valid_class_ids
    }

    groups = {name: [] for name in SPLIT_NAMES}
    remaining_capacity = capacities.copy()
    assigned_presence = {name: Counter() for name in SPLIT_NAMES}
    assigned_instances = {name: Counter() for name in SPLIT_NAMES}
    unassigned = set(range(len(records)))

    rng = random.Random(seed)
    candidate_ties = {index: rng.random() for index in range(len(records))}
    split_ties = {
        (index, name): rng.random()
        for index in range(len(records))
        for name in SPLIT_NAMES
    }

    def choose_candidate(focus_class: int) -> int:
        candidates = members[focus_class] & unassigned

        def candidate_score(index: int):
            record = records[index]
            scarcity = sum(
                1 / max(1, len(members[class_id] & unassigned))
                for class_id in record["labels"]
            )
            return (
                scarcity,
                len(record["labels"]),
                sum(record["instances"].values()),
                candidate_ties[index],
            )

        return max(candidates, key=candidate_score)

    def choose_split(index: int, focus_class: int) -> str:
        record = records[index]
        available = [name for name in SPLIT_NAMES if remaining_capacity[name] > 0]

        def split_score(name: str):
            focus_deficit = (
                presence_targets[focus_class][name]
                - assigned_presence[name][focus_class]
            )
            joint_deficit = sum(
                (
                    presence_targets[class_id][name]
                    - assigned_presence[name][class_id]
                ) / max(1, presence_totals[class_id])
                for class_id in record["labels"]
            )
            instance_deficit = sum(
                max(
                    0.0,
                    instance_targets[class_id][name]
                    - assigned_instances[name][class_id],
                ) / max(1, instance_totals[class_id])
                for class_id in record["labels"]
            )
            capacity_share = remaining_capacity[name] / max(1, capacities[name])
            return (
                focus_deficit,
                joint_deficit,
                instance_deficit,
                capacity_share,
                split_ties[(index, name)],
            )

        return max(available, key=split_score)

    while True:
        active_classes = [
            class_id for class_id in valid_class_ids
            if members[class_id] & unassigned
        ]
        if not active_classes:
            break
        focus_class = min(
            active_classes,
            key=lambda class_id: (len(members[class_id] & unassigned), class_id),
        )
        index = choose_candidate(focus_class)
        split_name = choose_split(index, focus_class)
        record = records[index]
        groups[split_name].append(record["item"])
        remaining_capacity[split_name] -= 1
        for class_id in record["labels"]:
            assigned_presence[split_name][class_id] += 1
            assigned_instances[split_name][class_id] += record["instances"][class_id]
        unassigned.remove(index)

    unlabelled = list(unassigned)
    rng.shuffle(unlabelled)
    for index in unlabelled:
        available = [name for name in SPLIT_NAMES if remaining_capacity[name] > 0]
        split_name = max(
            available,
            key=lambda name: (
                remaining_capacity[name] / max(1, capacities[name]),
                remaining_capacity[name],
                -SPLIT_NAMES.index(name),
            ),
        )
        groups[split_name].append(records[index]["item"])
        remaining_capacity[split_name] -= 1

    diagnostics = {
        "strategy": "multi_label_stratified",
        "seed": seed,
        "ratios": ratios,
        "capacities": capacities,
        "class_image_targets": presence_targets,
    }
    return groups, diagnostics
