"""Concrete model implementations — detectors, trackers, VLM backends.

Stub implementations (zero-dependency, always safe to import):
    StubDetector    — returns empty detection list
    StubTracker     — returns empty track list
    StubLocalVLM    — returns empty dict (LOCAL_MODEL)
    StubRemoteVLM   — returns empty dict (REMOTE_MODEL)

Real implementations (import their heavy deps lazily in load()/analyze_segment()):
    YOLODetector       — NOT YET IMPLEMENTED
    ByteTrackTracker   — NOT YET IMPLEMENTED
    LocalVLM           — NOT YET IMPLEMENTED (model TBD)
    RemoteVLM          — NOT YET IMPLEMENTED (provider TBD)
"""
from .stub_detector import StubDetector
from .stub_tracker import StubTracker
from .stub_vlm import StubLocalVLM, StubRemoteVLM

__all__ = [
    "StubDetector",
    "StubLocalVLM",
    "StubRemoteVLM",
    "StubTracker",
]
