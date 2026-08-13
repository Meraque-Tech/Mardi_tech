"""Regression tests for scalable Annotation QA persistence and paging."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from vision.ai.web.annotation_qa_store import (
    IssueWriter,
    filter_options,
    get_issue,
    import_issues,
    iter_legacy_issues,
    issue_count,
    query_issues,
    queue_counts,
    update_issue,
)


def issue(index: int, **updates) -> dict:
    payload = {
        "issue_id": f"job-{index:06d}",
        "image": f"/dataset/images/val/image-{index:06d}.jpg",
        "image_name": f"image-{index:06d}.jpg",
        "split": "val",
        "class_id": index % 3,
        "class_name": f"class-{index % 3}",
        "issue_type": "large_box_disagreement",
        "severity": "high" if index % 2 else "medium",
        "score": index / 1000,
        "review_status": "unreviewed",
        "accepted_fix": "",
        "accepted_class_id": None,
        "accepted_class_name": "",
        "audit_required": False,
        "audit_status": "not_required",
        "difference_band": "large_disagreement",
        "auto_fix_eligible": False,
        "quality_gate_passed": False,
        "fix_type": "",
        "recommended_bbox": None,
        "metrics": {},
    }
    payload.update(updates)
    return payload


def test_store_pages_and_filters_without_returning_the_full_report():
    with TemporaryDirectory() as directory:
        run_dir = Path(directory)
        writer = IssueWriter(run_dir, batch_size=37)
        for index in range(1200):
            writer.append(issue(index))
        writer.close()

        page = query_issues(run_dir, queue="all", page=7, page_size=50, class_name="class-1")

        assert issue_count(run_dir) == 1200
        assert page["total"] == 400
        assert page["page"] == 7
        assert len(page["items"]) == 50
        assert {item["class_name"] for item in page["items"]} == {"class-1"}
        assert filter_options(run_dir)["classes"] == ["class-0", "class-1", "class-2"]


def test_queue_counts_and_single_issue_updates_are_persistent():
    with TemporaryDirectory() as directory:
        run_dir = Path(directory)
        safe = issue(
            1,
            issue_type="moderate_box_difference",
            severity="low",
            difference_band="reviewable",
            auto_fix_eligible=True,
            quality_gate_passed=True,
            fix_type="replace_box",
            recommended_bbox=[1, 2, 3, 4],
            metrics={
                "prompt_stability": {"passed": True},
                "edge_differences": {"within_tolerance": False, "within_max_difference": True},
            },
        )
        audit = issue(2, audit_required=True, audit_status="pending")
        import_issues(run_dir, [safe, audit])

        before = queue_counts(run_dir)
        assert before["all"] == 2
        assert before["safe_suggestions"] == 1
        assert before["audits"] == 1

        updated = update_issue(run_dir, safe["issue_id"], lambda item: item.update(
            review_status="fix_accepted", accepted_fix="sam_box"
        ))

        assert updated["accepted_fix"] == "sam_box"
        assert get_issue(run_dir, safe["issue_id"])["review_status"] == "fix_accepted"
        after = queue_counts(run_dir)
        assert after["needs_review"] == 1
        assert after["resolved"] == 1
        assert after["accepted_total"] == 1


def test_payload_round_trips_nested_metrics():
    with TemporaryDirectory() as directory:
        run_dir = Path(directory)
        original = issue(9, metrics={"edge_differences": {"max_percent": 31.25}})
        import_issues(run_dir, [original])

        loaded = get_issue(run_dir, original["issue_id"])

        assert json.dumps(loaded, sort_keys=True) == json.dumps(original, sort_keys=True)


def test_legacy_import_skips_summary_issue_count_and_streams_top_level_array():
    with TemporaryDirectory() as directory:
        report_path = Path(directory) / "qa_report.json"
        expected = [issue(1), issue(2)]
        report_path.write_text(json.dumps({
            "summary": {"issues": 999, "splits": ["train", "val"]},
            "issues": expected,
        }), encoding="utf-8")

        assert list(iter_legacy_issues(report_path)) == expected
