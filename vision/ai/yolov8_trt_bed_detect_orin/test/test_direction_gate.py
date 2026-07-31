import sys
from pathlib import Path

import pytest


WEB_SERVER_DIR = Path(__file__).resolve().parents[1] / "web_server"
sys.path.insert(0, str(WEB_SERVER_DIR))

from direction_gate import SegmentFrameBuffer  # noqa: E402


def test_first_forward_segment_is_released_after_confirmation():
    frames = SegmentFrameBuffer(10)
    frames.add("startup-1")
    frames.add("startup-2")

    committed, discarded = frames.handle_state("forward")

    assert committed == ["startup-1", "startup-2"]
    assert discarded == 0
    assert len(frames) == 0


def test_first_reverse_candidate_segment_is_discarded():
    frames = SegmentFrameBuffer(10)
    frames.add("reverse-candidate")

    committed, discarded = frames.handle_state("reverse_suspected")

    assert committed == []
    assert discarded == 1
    assert len(frames) == 0


def test_reversing_segments_are_discarded():
    frames = SegmentFrameBuffer(10)
    frames.add("reverse-confirmation")

    committed, discarded = frames.handle_state("reversing")

    assert committed == []
    assert discarded == 1


def test_forward_candidate_is_held_until_second_confirmation():
    frames = SegmentFrameBuffer(10)
    frames.add("forward-candidate")

    committed, discarded = frames.handle_state("forward_suspected")

    assert committed == []
    assert discarded == 0
    assert len(frames) == 1

    frames.add("forward-confirmation")
    committed, discarded = frames.handle_state("forward")

    assert committed == ["forward-candidate", "forward-confirmation"]
    assert discarded == 0


def test_failed_forward_candidate_is_discarded_on_reversing_state():
    frames = SegmentFrameBuffer(10)
    frames.add("false-forward-candidate")
    frames.handle_state("forward_suspected")

    committed, discarded = frames.handle_state("reversing")

    assert committed == []
    assert discarded == 1


def test_stationary_and_initializing_states_discard_unclassified_frames():
    frames = SegmentFrameBuffer(10)
    frames.add("stationary")
    assert frames.handle_state("stationary") == ([], 1)

    frames.add("initializing")
    assert frames.handle_state("initializing") == ([], 1)


def test_capacity_discards_oldest_frame_without_writing_it():
    frames = SegmentFrameBuffer(2)

    assert frames.add("oldest") is None
    assert frames.add("middle") is None
    assert frames.add("newest") == "oldest"
    assert frames.handle_state("forward") == (["middle", "newest"], 0)


def test_invalid_configuration_and_state_are_rejected():
    with pytest.raises(ValueError):
        SegmentFrameBuffer(0)

    with pytest.raises(ValueError):
        SegmentFrameBuffer(1).handle_state("sideways")
