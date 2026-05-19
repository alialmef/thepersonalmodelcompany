"""Tests for the founder tracker (first-100-users free)."""

from __future__ import annotations

from pathlib import Path

from pmc.storage.founders import FounderTracker


def test_initial_state_full_slots(tmp_path: Path):
    tracker = FounderTracker(tmp_path)
    assert tracker.used() == 0
    assert tracker.slots_remaining() == 100


def test_grant_marks_user_as_founder(tmp_path: Path):
    tracker = FounderTracker(tmp_path)
    grant = tracker.grant_if_available("user-1")
    assert grant.is_founder is True
    assert grant.granted_now is True
    assert grant.slots_remaining == 99
    assert grant.granted_at is not None
    assert tracker.is_founder("user-1")
    assert tracker.used() == 1


def test_grant_idempotent(tmp_path: Path):
    tracker = FounderTracker(tmp_path)
    g1 = tracker.grant_if_available("alex")
    g2 = tracker.grant_if_available("alex")
    assert g1.granted_now is True
    assert g2.granted_now is False
    assert g2.is_founder is True
    assert tracker.used() == 1  # only counts once


def test_grant_denied_when_slots_exhausted(tmp_path: Path):
    tracker = FounderTracker(tmp_path, total_slots=3)
    for uid in ("a", "b", "c"):
        grant = tracker.grant_if_available(uid)
        assert grant.is_founder is True
    # Fourth user is past the cap
    grant4 = tracker.grant_if_available("d")
    assert grant4.is_founder is False
    assert grant4.slots_remaining == 0
    assert tracker.is_founder("d") is False
    assert tracker.used() == 3


def test_state_persists_across_instances(tmp_path: Path):
    t1 = FounderTracker(tmp_path)
    t1.grant_if_available("user-1")
    t1.grant_if_available("user-2")
    # New tracker instance, same root → reads the persisted state
    t2 = FounderTracker(tmp_path)
    assert t2.used() == 2
    assert t2.is_founder("user-1")
    assert t2.is_founder("user-2")
    assert t2.slots_remaining() == 98


def test_existing_founder_unaffected_by_cap_change(tmp_path: Path):
    """If we lower the total_slots later, existing founders still count as founders."""
    t1 = FounderTracker(tmp_path, total_slots=100)
    for i in range(5):
        t1.grant_if_available(f"user-{i}")

    # Now restrict to 3 — existing founders stay, but new grants honor the new cap
    t2 = FounderTracker(tmp_path, total_slots=3)
    assert t2.is_founder("user-0")
    assert t2.is_founder("user-4")
    # No new grants possible — used (5) > total (3)
    grant = t2.grant_if_available("newcomer")
    assert grant.is_founder is False


def test_slots_remaining_never_negative(tmp_path: Path):
    tracker = FounderTracker(tmp_path, total_slots=2)
    tracker.grant_if_available("a")
    tracker.grant_if_available("b")
    # used == total; remaining should be 0, not negative
    assert tracker.slots_remaining() == 0


def test_state_serialization_roundtrip(tmp_path: Path):
    tracker = FounderTracker(tmp_path, total_slots=50)
    tracker.grant_if_available("alpha")
    tracker.grant_if_available("beta")
    state = tracker.state()
    assert state.total_slots == 50
    assert state.used == 2
    assert "alpha" in state.founders
    assert "beta" in state.founders
    assert "alpha" in state.granted_at


def test_first_100_users_get_free(tmp_path: Path):
    """The named feature: 100 users get founder status, 101st doesn't."""
    tracker = FounderTracker(tmp_path)  # default 100 slots
    for i in range(100):
        grant = tracker.grant_if_available(f"u-{i:03d}")
        assert grant.is_founder, f"user {i} should be a founder"
    grant101 = tracker.grant_if_available("u-100")
    assert grant101.is_founder is False
    assert grant101.slots_remaining == 0
    assert tracker.used() == 100
