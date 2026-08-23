
from src.config import EpisodeConfig
from src.models.episode_assembler import EpisodeAssembler
from src.schema.event import PhysicalEvent


def create_event(eid, start, end, actor, obj, action):
    return PhysicalEvent(
        event_id=eid,
        start_sec=start,
        end_sec=end,
        actor_track_id=actor,
        object_track_id=obj,
        action=action,
        review_status="PENDING",
        attributes={"timing_precision": "EXACT"},
        source="test"
    )

def test_assembler():
    config = EpisodeConfig(enabled=True, max_event_gap_sec=2.0, require_shared_entity=True)
    assembler = EpisodeAssembler(config)

    # Bridge case: PICK Cup 17 (1-2s) + TOUCH Table (2-3s) + PLACE Cup 17 (3-4s)
    events = [
        create_event("e1", 1.0, 2.0, 1, 17, "PICK"),
        create_event("e2", 2.0, 3.0, 1, 1, "TOUCH"),
        create_event("e3", 3.0, 4.0, 1, 17, "PLACE"),
    ]

    episodes = assembler.assemble(events)
    assert len(episodes) == 2, f"Expected 2 episodes, got {len(episodes)}"
    
    # Check that e1 and e3 are together, and e2 is separate
    e1_e3 = next(ep for ep in episodes if "e1" in ep.event_ids)
    assert "e3" in e1_e3.event_ids
    assert "e2" not in e1_e3.event_ids

    e2_ep = next(ep for ep in episodes if "e2" in ep.event_ids)
    assert len(e2_ep.event_ids) == 1

    print("Smoke test passed!")

if __name__ == "__main__":
    test_assembler()
