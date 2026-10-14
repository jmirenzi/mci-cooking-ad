"""Download the EPIC-Tent annotations. No video: 16 MB of text, not the 166 GB release.

The HSMM and the LLM arms consume symbolic (verb, noun) tick streams, so everything this project
needs is in two CSVs published separately from the video, on the authors' annotation repo. The
data.bris release (166.8 GiB) carries the GoPro/SMI video, the gaze traces and the per-frame
uncertainty ratings; only the last of those is worth fetching, and only for one purpose (see
--uncertainty).

Row counts are asserted rather than trusted. A truncated download that still parses is the
failure mode worth catching -- the same reason parse_breakfast._check_against_config raises.

Licence: CC BY-NC 4.0 (annotations) and a Non-Commercial Government Licence (the data.bris
release). Cite Jang et al., ICCVW 2019.

    ./py scripts/data/fetch_epic_tent.py --out dataset/epic_tent
    ./py scripts/data/fetch_epic_tent.py --out dataset/epic_tent --uncertainty
"""
import argparse
import logging
import urllib.error
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

ANNOTATION_BASE = "https://raw.githubusercontent.com/youngkyoonjang/EPIC_Tent2019/master/"

# data.bris serves this prefix as an open directory index, so the per-participant uncertainty
# files can be fetched individually instead of pulling the whole release.
RELEASE_BASE = "https://data.bris.ac.uk/datasets/2ite3tu1u53n42hjfh3886sa86/"

SUBJECTS = range(1, 30)

# Expected data rows (excluding the header). Measured against the Oct 2019 release; a mismatch
# means either a truncated download or a re-release, and both need a human.
EXPECTED_ROWS = {"Synchronised_action_label.txt": 1261, "Synchronised_error_label.txt": 626}

# Participants 1-7 rated no uncertainty -- the procedure was added partway through collection.
UNCERTAINTY_SUBJECTS = range(8, 30)


def _get(url):
    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"{url} -> HTTP {exc.code}") from exc


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    logger.info("%s (%.1f KiB)", path, len(payload) / 1024)


def fetch_annotations(out_dir):
    """The two CSVs this project actually parses, plus the README that documents their columns."""
    for name in (*EXPECTED_ROWS, "README.md", "load_synchronised_action_error_annotations.py"):
        payload = _get(ANNOTATION_BASE + name)
        expected = EXPECTED_ROWS.get(name)
        if expected is not None:
            rows = len([line for line in payload.decode().splitlines() if line.strip()]) - 1
            if rows != expected:
                raise RuntimeError(f"{name}: {rows} data rows, expected {expected}")
        _write(out_dir / name, payload)


def fetch_frame_level(out_dir):
    """Per-frame action ids, one file per participant. Unused by the tick ingest -- the segment
    CSV is the same information in 1/300th the bytes -- but it is the only per-frame record, so
    it is worth having locally before checking a disagreement against the segments."""
    for subject in SUBJECTS:
        name = f"{subject:02d}.frame_level_gt.txt"
        _write(out_dir / "frame_level_action_annotation" / name,
               _get(f"{ANNOTATION_BASE}frame_level_action_annotation/{name}"))


def fetch_uncertainty(out_dir):
    """Per-frame self-rated uncertainty, 0 (certain) to 1. The only use here is PREGO's
    train/test split: test = median confidence (1 - uncertainty) under 0.6, plus the 7
    participants with no rating (supplementary C). Note the row counts do not all match the
    release README's frame counts (participants 14, 26, 28, 29 disagree); PREGO used the same
    files, so scripts/data/epic_tent_splits.py uses them as shipped and reports the mismatch."""
    for subject in UNCERTAINTY_SUBJECTS:
        name = f"{subject:02d}.uncertainty.gopro_timecode.txt"
        _write(out_dir / "uncertainty" / name,
               _get(f"{RELEASE_BASE}data/{subject:02d}/{name}"))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="dataset/epic_tent")
    ap.add_argument("--uncertainty", action="store_true",
                    help="also fetch the per-frame uncertainty ratings (~22 MB, participants "
                         "8-29 only) -- needed only for the PREGO-comparable split")
    ap.add_argument("--skip-frame-level", action="store_true",
                    help="skip the per-frame action ids (~16 MB), which the tick ingest does "
                         "not read")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out_dir = Path(args.out)
    fetch_annotations(out_dir)
    if not args.skip_frame_level:
        fetch_frame_level(out_dir)
    if args.uncertainty:
        fetch_uncertainty(out_dir)
    logger.info("annotations in %s", out_dir)


if __name__ == "__main__":
    main()
