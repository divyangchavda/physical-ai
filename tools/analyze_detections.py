import json
import sys

det_file = sys.argv[1]
with open(det_file, 'r') as f:
    data = json.load(f)

total_frames = len(data)
total_detections = sum(len(f["detections"]) for f in data)

classes = {}
for frame in data:
    for det in frame["detections"]:
        cn = det['class_name']
        classes[cn] = classes.get(cn, 0) + 1

print(f'Total frames: {total_frames}')
print(f'Total detections: {total_detections}')
print(f'Classes detected:')
for cls, count in sorted(classes.items(), key=lambda x: -x[1]):
    print(f'  {cls}: {count}')
