import json
import sys

y8_file = sys.argv[1]
yw_file = sys.argv[2]

with open(y8_file, 'r') as f:
    y8_data = json.load(f)
with open(yw_file, 'r') as f:
    yw_data = json.load(f)

print("=" * 70)
print("TT6 COMPARISON: YOLOv8n vs YOLO-World")
print("=" * 70)

# Detections
y8_total = sum(len(f["detections"]) for f in y8_data)
yw_total = sum(len(f["detections"]) for f in yw_data)

print(f"\nDETECTIONS:")
print(f"  YOLOv8n:     {y8_total} detections across {len(y8_data)} frames ({y8_total/len(y8_data):.2f} per frame)")
print(f"  YOLO-World:  {yw_total} detections across {len(yw_data)} frames ({yw_total/len(yw_data):.2f} per frame)")
print(f"  Difference:  {yw_total - y8_total:+d} ({(yw_total/y8_total - 1)*100:+.1f}%)")

# Classes
y8_classes = {}
yw_classes = {}
for frame in y8_data:
    for det in frame["detections"]:
        cn = det['class_name']
        y8_classes[cn] = y8_classes.get(cn, 0) + 1
for frame in yw_data:
    for det in frame["detections"]:
        cn = det['class_name']
        yw_classes[cn] = yw_classes.get(cn, 0) + 1

print(f"\nCLASS BREAKDOWN:")
all_classes = sorted(set(list(y8_classes.keys()) + list(yw_classes.keys())))
print(f"  {'Class':<20} {'YOLOv8n':>10} {'YOLO-World':>12} {'Diff':>8}")
print(f"  {'-'*20} {'-'*10} {'-'*12} {'-'*8}")
for cls in all_classes:
    y8_c = y8_classes.get(cls, 0)
    yw_c = yw_classes.get(cls, 0)
    diff = yw_c - y8_c
    print(f"  {cls:<20} {y8_c:>10} {yw_c:>12} {diff:>+8}")

# Tracks
with open(y8_file.replace('detections.json', 'tracks.json'), 'r') as f:
    y8_tracks = json.load(f)
with open(yw_file.replace('detections.json', 'tracks.json'), 'r') as f:
    yw_tracks = json.load(f)

print(f"\nTRACKS:")
print(f"  YOLOv8n:     {len(y8_tracks)} unique tracks")
print(f"  YOLO-World:  {len(yw_tracks)} unique tracks")
print(f"  Difference:  {len(yw_tracks) - len(y8_tracks):+d}")

# Track details
print(f"\nTRACK BREAKDOWN:")
print(f"  {'Backend':<12} {'Track ID':>10} {'Class':<20} {'Points':>8} {'Duration':>12}")
print(f"  {'-'*12} {'-'*10} {'-'*20} {'-'*8} {'-'*12}")
for t in y8_tracks:
    duration = f"{t['points'][0]['timestamp_sec']:.1f}-{t['points'][-1]['timestamp_sec']:.1f}s"
    print(f"  {'YOLOv8n':<12} {t['track_id']:>10} {t['class_name']:<20} {len(t['points']):>8} {duration:>12}")
for t in yw_tracks:
    duration = f"{t['points'][0]['timestamp_sec']:.1f}-{t['points'][-1]['timestamp_sec']:.1f}s"
    print(f"  {'YOLO-World':<12} {t['track_id']:>10} {t['class_name']:<20} {len(t['points']):>8} {duration:>12}")

print("\n" + "=" * 70)
