"""Build a small, balanced, labelled evaluation bundle out of a local SSv2 dump.

Why this exists: every number this project has is measured on one 6.7-second video
with seven hand-written labels, where a uniform grid of times scores 8/8 against
our geometry's 7/8. That clip cannot distinguish a real improvement from luck.
SSv2 gives 220,847 clips with one human-verified action label each, and one clip is
one action covering the whole clip, so there is no timing ambiguity to argue about.

What it does NOT do: run anything. No model, no GPU, no network. It reads the
labels, joins them to the clips already on disk, keeps the ones our vocabulary can
be judged on, and copies a few hundred into an upload folder with their truth.

Sampling is **balanced per verb, not per class**. Twelve SSv2 classes map to PLACE
and one maps to TOUCH, so a uniform sample over clips would be roughly a PLACE
accuracy measurement wearing a coat. Equal clips per verb is a choice, stated here:
it makes the per-verb columns comparable and makes the overall figure a mean over
verbs rather than a mean over SSv2's class frequencies.

Selection is deterministic — clips are taken in sorted-id order, no RNG — so the
same disk produces the same bundle, and a rerun after a code change compares
like with like. The archive itself stores clips in shuffled order, so a sorted-id
prefix is still spread across the dataset rather than being one contiguous chunk
of it.

Usage (nothing here is Kaggle-specific):

    python tools/ssv2_bundle.py \
        --clips  D:/ddd/ssv2_subset/20bn-something-something-v2 \
        --labels D:/ddd/lables/labels \
        --out    D:/ddd/ssv2_upload \
        --limit  200
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval.ssv2_action_map import (  # noqa: E402
    SSV2_TEMPLATE_TO_ACTION,
    map_template,
    normalize_template,
)

# The two splits that ship template + placeholders per clip. test.json carries ids
# only and test-answers.csv carries the label sentence without the template or the
# object list, so the test split cannot be used here at all.
LABELLED_SPLITS = ("train.json", "validation.json")


def verify_allow_list(labels_dir: Path) -> None:
    """Fail if any allow-list key is not a real SSv2 class name.

    The 43 keys were typed from the class list by hand. A typo would silently
    shrink the evaluation set instead of erroring, and a shrunken set is the kind
    of thing that gets read as "this verb never appears" later on. So every key is
    checked against the dataset's own labels.json before anything is copied.
    """
    catalogue = json.loads((labels_dir / "labels.json").read_text(encoding="utf-8"))
    known = {normalize_template(name) for name in catalogue}
    missing = sorted(
        name for name in SSV2_TEMPLATE_TO_ACTION
        if normalize_template(name) not in known
    )
    if missing:
        raise SystemExit(
            f"{len(missing)} allow-list key(s) are not SSv2 class names:\n  "
            + "\n  ".join(missing)
        )
    print(f"allow-list: {len(SSV2_TEMPLATE_TO_ACTION)} of {len(catalogue)} "
          f"SSv2 classes, all names verified against labels.json")


def load_truth(labels_dir: Path) -> dict[str, dict]:
    """Every labelled clip, keyed by id, from the splits that carry a template."""
    out: dict[str, dict] = {}
    for split in LABELLED_SPLITS:
        path = labels_dir / split
        entries = json.loads(path.read_text(encoding="utf-8"))
        for entry in entries:
            out[str(entry["id"])] = {**entry, "split": split.removesuffix(".json")}
        print(f"{split}: {len(entries)} labelled clips")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", required=True, type=Path,
                        help="directory holding the extracted <id>.webm files")
    parser.add_argument("--labels", required=True, type=Path,
                        help="directory holding labels.json / train.json / validation.json")
    parser.add_argument("--out", required=True, type=Path,
                        help="bundle directory to create")
    parser.add_argument("--limit", type=int, default=200,
                        help="total clips to copy, split as evenly as possible "
                             "across the verbs actually available (default 200)")
    args = parser.parse_args()

    verify_allow_list(args.labels)
    truth = load_truth(args.labels)

    on_disk = sorted(p for p in args.clips.glob("*.webm"))
    print(f"clips on disk: {len(on_disk)}")
    if not on_disk:
        raise SystemExit(f"no .webm files under {args.clips}")

    # Join disk to labels, then to the allow-list. Both drops are counted rather
    # than passed over: "how much of a random SSv2 sample can we even be scored
    # on" is a fact about our vocabulary and is worth printing.
    unlabelled = 0
    excluded: dict[str, int] = defaultdict(int)
    by_verb: dict[str, list[dict]] = defaultdict(list)
    for path in on_disk:
        entry = truth.get(path.stem)
        if entry is None:
            unlabelled += 1
            continue
        action = map_template(entry["template"])
        if action is None:
            excluded[entry["template"]] += 1
            continue
        by_verb[action.value].append({
            "clip_id": path.stem,
            "path": path,
            "action": action.value,
            "template": entry["template"],
            "label": entry["label"],
            "placeholders": entry["placeholders"],
            "split": entry["split"],
        })

    usable = sum(len(v) for v in by_verb.values())
    print(f"  unlabelled (test split): {unlabelled}")
    print(f"  excluded by vocabulary : {sum(excluded.values())} clips "
          f"over {len(excluded)} classes")
    print(f"  usable                 : {usable} clips over {len(by_verb)} verbs")

    # Balanced draw. Verbs with fewer clips than their share give the remainder
    # back to the others, so --limit is met whenever the disk can meet it.
    verbs = sorted(by_verb)
    for entries in by_verb.values():
        entries.sort(key=lambda e: e["clip_id"])
    chosen: list[dict] = []
    remaining_verbs = list(verbs)
    taken = {verb: 0 for verb in verbs}
    while remaining_verbs and len(chosen) < args.limit:
        share = max(1, (args.limit - len(chosen)) // len(remaining_verbs))
        for verb in list(remaining_verbs):
            pool = by_verb[verb]
            room = min(share, len(pool) - taken[verb], args.limit - len(chosen))
            if room <= 0:
                remaining_verbs.remove(verb)
                continue
            chosen.extend(pool[taken[verb]:taken[verb] + room])
            taken[verb] += room
            if taken[verb] >= len(pool):
                remaining_verbs.remove(verb)

    print("\nper-verb draw (taken / available):")
    for verb in verbs:
        print(f"  {verb:<8} {taken[verb]:>4} / {len(by_verb[verb]):>5}")

    clips_dir = args.out / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    total_bytes = 0
    for entry in chosen:
        dest = clips_dir / f"{entry['clip_id']}.webm"
        shutil.copy2(entry["path"], dest)
        total_bytes += dest.stat().st_size

    manifest = {
        "_source": "Something-Something V2, part 00 of the Qualcomm TGZ, "
                   "extracted locally; labels from the official label package",
        "_what": "One human-verified action label per clip. One clip is one "
                 "action covering the whole clip, so there is no timing "
                 "ambiguity: a verb is right or it is not.",
        "_sampling": "Balanced per ActionType, deterministic (sorted clip id). "
                     "See tools/ssv2_bundle.py for why balance is per verb.",
        "_not_measured": [
            "object detection -- text_prompt is built from the label's own "
            "placeholders, so the object names are given, not found",
            "timing -- SSv2 has no timestamps; every clip's action spans it",
        ],
        "_counts": {
            "clips_on_disk_considered": len(on_disk),
            "unlabelled_test_split": unlabelled,
            "excluded_by_vocabulary": sum(excluded.values()),
            "usable": usable,
            "bundled": len(chosen),
        },
        "clips": [
            {k: v for k, v in entry.items() if k != "path"} for entry in chosen
        ],
    }
    (args.out / "truth.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(f"\nbundled {len(chosen)} clips -> {clips_dir} "
          f"({total_bytes / 1e6:.1f} MB)")
    print(f"truth -> {args.out / 'truth.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
