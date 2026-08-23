import sys
from pathlib import Path
import json

from tests.test_phase1_ingest_sample import make_video, make_pipeline_context
from src.stages import s01_ingest, s02_sample
from src.logging_utils import get_logger

logger = get_logger(__name__)

def run_simulated_real_test():
    workspace = Path(".").resolve()
    video_path = workspace / "input" / "test_10s.mp4"
    out_dir = workspace / "output_test"
    
    video_path.parent.mkdir(exist_ok=True)
    out_dir.mkdir(exist_ok=True)
    
    # 1. Create a 10s video at 30 FPS
    print(f"Creating 10s synthetic video at {video_path}")
    make_video(video_path, duration_sec=10.0, fps=30.0)
    
    # 2. Run s01 and s02
    print("Running ingest and sample stages...")
    ctx = make_pipeline_context(video_path, out_dir, fps=1.0)
    s01_stat = s01_ingest.run(ctx)
    s02_stat = s02_sample.run(ctx)
    
    print("\n--- RESULTS ---")
    print(f"s01 status: {s01_stat.status} | msg: {s01_stat.message}")
    print(f"s02 status: {s02_stat.status} | msg: {s02_stat.message}")
    
    vm = ctx.video_metadata
    print(f"\nMetadata: {vm.duration_sec:.2f}s | {vm.fps:.2f} fps | {vm.width}x{vm.height} | frames: {vm.frame_count} | codec: {vm.codec}")
    
    n_samples = len(ctx.sampled_frame_infos)
    print(f"Sampled frames: {n_samples}")
    
    if n_samples > 0:
        first = ctx.sampled_frame_infos[0]
        last = ctx.sampled_frame_infos[-1]
        print(f"First sample: idx {first.frame_index} @ {first.timestamp_sec:.3f}s")
        print(f"Last sample:  idx {last.frame_index} @ {last.timestamp_sec:.3f}s")
        
    print("\nCheck output_test/sampling_plan.json for full plan.")

if __name__ == "__main__":
    run_simulated_real_test()
