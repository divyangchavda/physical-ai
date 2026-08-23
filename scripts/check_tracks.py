import json

try:
    with open('output/tracks.json') as f:
        tracks = json.load(f)
    for t in tracks:
        print(f"Track {t['track_id']}: {t['class_name']} ({len(t['points'])} pts)")
except Exception as e:
    print(e)
