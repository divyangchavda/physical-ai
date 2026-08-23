import hashlib

from src.config import EpisodeConfig
from src.schema.episode import InteractionEpisode
from src.schema.event import PhysicalEvent


class EpisodeAssembler:
    def __init__(self, config: EpisodeConfig):
        self.config = config

    def assemble(self, events: list[PhysicalEvent]) -> list[InteractionEpisode]:
        valid_events = [e for e in events if e.review_status != "REJECT"]
        valid_events.sort(key=lambda e: (e.start_sec, e.end_sec, e.event_id))

        active_episodes: list[dict] = []
        completed_episodes: list[dict] = []

        def _compute_quality(q1: str | None, q2: str | None) -> str | None:
            if q1 is None or q2 is None:
                return q1 or q2
            order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
            return q1 if order.get(q1, 0) < order.get(q2, 0) else q2

        def _merge(episode_data: dict, event: PhysicalEvent, reason: str):
            episode_data["events"].append(event)
            episode_data["end_sec"] = max(episode_data["end_sec"], event.end_sec)
            
            p1 = episode_data["timing_precision"]
            # Just default to EXACT unless attributes says otherwise for now
            p2 = event.attributes.get("timing_precision", "EXACT")
            if p1 == p2:
                episode_data["timing_precision"] = p1
            else:
                episode_data["timing_precision"] = "MIXED"
                
            if event.actor_track_id is not None:
                aid = str(event.actor_track_id)
                if aid not in episode_data["actor_track_ids"]:
                    episode_data["actor_track_ids"].append(aid)
            if event.object_track_id is not None:
                oid = str(event.object_track_id)
                if oid not in episode_data["object_track_ids"]:
                    episode_data["object_track_ids"].append(oid)

            # Extra ids from attributes
            ge_id = event.attributes.get("graph_edge_id")
            if ge_id and ge_id not in episode_data["graph_edge_ids"]:
                episode_data["graph_edge_ids"].append(ge_id)
            st_id = event.attributes.get("state_transition_id")
            if st_id and st_id not in episode_data["state_transition_ids"]:
                episode_data["state_transition_ids"].append(st_id)
            qs_id = event.attributes.get("quality_score_id")
            if qs_id and qs_id not in episode_data["quality_score_ids"]:
                episode_data["quality_score_ids"].append(qs_id)
                
            if event.observation_id and event.observation_id not in episode_data["observation_ids"]:
                episode_data["observation_ids"].append(event.observation_id)
                    
            event_quality = event.attributes.get("event_quality_tier", "HIGH")
            episode_data["episode_quality_tier"] = _compute_quality(
                episode_data["episode_quality_tier"], event_quality
            )
            episode_data["is_estimated"] = episode_data["is_estimated"] or event.is_estimated
            if reason not in episode_data["assembly_reasons"]:
                episode_data["assembly_reasons"].append(reason)

        for event in valid_events:
            best_match = None
            best_reason = ""
            
            for ep in active_episodes:
                gap = event.start_sec - ep["end_sec"]
                if gap > self.config.max_event_gap_sec:
                    continue
                    
                if not self.config.require_shared_entity:
                    best_match = ep
                    best_reason = "temporal"
                    break
                    
                # Check semantic anchor
                ev_obj = str(event.object_track_id) if event.object_track_id is not None else None
                ev_act = str(event.actor_track_id) if event.actor_track_id is not None else None
                
                shared_object = False
                if ev_obj and ev_obj in ep["object_track_ids"] and not ev_obj.startswith("-1"):
                    shared_object = True
                    
                shared_actor = False
                if ev_act and ev_act in ep["actor_track_ids"] and not ev_act.startswith("-1"):
                    shared_actor = True
                
                if shared_object:
                    best_match = ep
                    best_reason = "shared_object"
                    break
                    
                if shared_actor:
                    ev_obj_valid = ev_obj and not ev_obj.startswith("-1")
                    ep_objs = [o for o in ep["object_track_ids"] if not o.startswith("-1")]
                    
                    if ev_obj_valid and ep_objs and ev_obj not in ep_objs:
                        continue
                    
                    best_match = ep
                    best_reason = "shared_actor"
                    break

            if best_match:
                _merge(best_match, event, best_reason)
            else:
                ge_id = event.attributes.get("graph_edge_id")
                st_id = event.attributes.get("state_transition_id")
                qs_id = event.attributes.get("quality_score_id")
                event_quality = event.attributes.get("event_quality_tier", "HIGH")
                new_ep = {
                    "events": [event],
                    "start_sec": event.start_sec,
                    "end_sec": event.end_sec,
                    "timing_precision": event.attributes.get("timing_precision", "EXACT"),
                    "actor_track_ids": [str(event.actor_track_id)] if event.actor_track_id is not None else [],
                    "object_track_ids": [str(event.object_track_id)] if event.object_track_id is not None else [],
                    "graph_edge_ids": [ge_id] if ge_id else [],
                    "state_transition_ids": [st_id] if st_id else [],
                    "quality_score_ids": [qs_id] if qs_id else [],
                    "observation_ids": [event.observation_id] if event.observation_id else [],
                    "episode_quality_tier": event_quality,
                    "is_estimated": event.is_estimated,
                    "assembly_reasons": ["initial"]
                }
                active_episodes.append(new_ep)
                
            still_active = []
            for ep in active_episodes:
                if event.start_sec - ep["end_sec"] > self.config.max_event_gap_sec:
                    completed_episodes.append(ep)
                else:
                    still_active.append(ep)
            active_episodes = still_active

        completed_episodes.extend(active_episodes)
        
        result = []
        for ep_data in completed_episodes:
            event_ids = [e.event_id for e in ep_data["events"]]
            hash_input = "-".join(event_ids)
            stable_hash = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()[:8]
            result.append(InteractionEpisode(
                episode_id=f"ie_{stable_hash}",
                event_ids=event_ids,
                start_sec=ep_data["start_sec"],
                end_sec=ep_data["end_sec"],
                timing_precision=ep_data["timing_precision"],
                actor_track_ids=ep_data["actor_track_ids"],
                object_track_ids=ep_data["object_track_ids"],
                graph_edge_ids=ep_data["graph_edge_ids"],
                state_transition_ids=ep_data["state_transition_ids"],
                quality_score_ids=ep_data["quality_score_ids"],
                observation_ids=ep_data["observation_ids"],
                episode_quality_tier=ep_data["episode_quality_tier"],
                source="EpisodeAssembler",
                is_estimated=ep_data["is_estimated"],
                assembly_reasons=ep_data["assembly_reasons"],
            ))
            
        result.sort(key=lambda r: r.start_sec)
        return result
