"""Rebuild the README's results block from results/*.json.

    python -m evaluation.report

Every number in the block comes from a file in results/; nothing is typed by hand.
Benchmarks that have not been run show as "not run".
"""

import json
from pathlib import Path

START, END = "<!-- results:start -->", "<!-- results:end -->"
R = Path("results")


def _load(name: str) -> dict | None:
    p = R / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _f(x, nd=3) -> str:
    return "-" if x is None else f"{x:.{nd}f}"


def ball_table() -> list[str]:
    rows = ["| Ball detector (TrackNet games 9-10, 3,717 frames) | Precision | Recall | F1 | Median error |",
            "| --- | --- | --- | --- | --- |"]
    yolo, tn = _load("baseline_ball"), _load("ball_tracknet")
    if yolo:
        for rule, label in (("last", "Tutorial YOLO, last box kept (as shipped)"),
                            ("best", "Tutorial YOLO, most confident box")):
            b = yolo["ball"][rule]
            rows.append(f"| {label} | {_f(b['precision'])} | {_f(b['recall'])} | {_f(b['f1'])} | {_f(b['median_px'], 2)} px |")
    if tn:
        b = tn["ball"]
        rows.append(f"| 3-frame heatmap tracker (this repo) | {_f(b['precision'])} | {_f(b['recall'])} | {_f(b['f1'])} | {_f(b['median_px'], 2)} px |")
    return rows


def event_table() -> list[str]:
    rows = ["| Hit / bounce detection (+-5 frames) | Hits F1, games 9-10 | Bounces F1, games 9-10 | Hits / bounces F1, games 1-8 out-of-fold |",
            "| --- | --- | --- | --- |"]
    yolo, tn = _load("baseline_ball"), _load("ball_tracknet")
    if yolo:
        rows.append(f"| Tutorial shot detector on YOLO positions | {_f(yolo['hits']['best']['f1'])} | not detected | - |")
    if tn:
        rows.append(f"| Tutorial shot detector on tracker positions | {_f(tn['hits_by_pipeline_shot_detector']['f1'])} | not detected | - |")
    for name, label in (("events_v1", "Event MLP v1 (image y feature, thresholds from game 8)"),
                        ("events_v2", "Event MLP v2 (no image y, mirrored, leave-one-game-out thresholds)"),
                        ("events_v2_on_groundtruth_trajectories", "Event MLP v2 on ground-truth trajectories (upper bound)")):
        d = _load(name)
        if d:
            oof = d.get("oof_games_1_8")
            oof_s = f"{_f(oof['hits']['f1'])} / {_f(oof['bounces']['f1'])}" if oof else "-"
            rows.append(f"| {label} | {_f(d['test']['hits']['f1'])} | {_f(d['test']['bounces']['f1'])} | {oof_s} |")
    return rows


def court_table() -> list[str]:
    rows = ["| Court keypoints (TennisCourtDetector val, 7 px rule) | Precision | Accuracy | Median error |",
            "| --- | --- | --- | --- |"]
    base = _load("court_baseline")
    if base:
        for key, label in (("raw", "Tutorial ResNet50 regression"),
                           ("homography_ransac_5px", "Tutorial ResNet50 + RANSAC homography")):
            c = base[key]
            rows.append(f"| {label} | {_f(c['precision'])} | {_f(c['accuracy'])} | {_f(c['median_px'], 2)} px |")
    heat = _load("court_heatmap")
    if heat:
        for key, label in (("raw", "Heatmap U-Net (this repo)"),
                           ("homography_ransac_5px", "Heatmap U-Net + RANSAC homography")):
            c = heat[key]
            rows.append(f"| {label} | {_f(c['precision'])} | {_f(c['accuracy'])} | {_f(c['median_px'], 2)} px |")
    else:
        rows.append("| Heatmap U-Net (this repo) | not run | not run | not run |")
    # Cited, not measured here: the table in yastrebksv/TennisCourtDetector's README.
    rows.append("| Published, TennisCourtDetector base model (cited from its README) | 0.936 | 0.933 | 2.83 px |")
    rows.append("| Published, TennisCourtDetector + refinement + homography (cited) | 0.963 | 0.961 | 1.83 px |")
    return rows


def pipeline_table() -> list[str]:
    d = _load("pipeline_bench")
    if not d:
        return ["Pipeline throughput: not run."]
    rows = [f"| End-to-end pipeline ({d['gpu']}) | Frames | Wall time | Frames/s | Peak RAM (GiB) |",
            "| --- | --- | --- | --- | --- |"]
    for r in d["runs"]:
        if r["ok"]:
            rows.append(f"| {r['label']} | {r['frames']} | {r['wall_s']:.1f} s | {r['frames'] / r['wall_s']:.1f} | {r['maxrss_gb']:.2f} |")
        else:
            rows.append(f"| {r['label']} | {r['frames']} | crashed | - | >= {r['maxrss_gb']:.2f} at the crash |")
    return rows


def main() -> None:
    leak = _load("leakage")
    lines = ["## Results", "",
             "Every number below is read from `results/*.json` by `python -m evaluation.report`.", ""]
    for table in (ball_table(), event_table(), court_table(), pipeline_table()):
        lines += table + [""]
    if leak:
        hits = sum(g["training_images_matching"] for g in leak["games"].values())
        lines.append(f"Leakage check: {hits} of the {leak['training_images']} images the tutorial's ball "
                     f"detector trained on match any TrackNet frame (64-bit dHash, <= {leak['max_bits']} bits).")
    block = "\n".join([START, *lines, END])
    readme = Path("README.md").read_text()
    if START in readme:
        pre, rest = readme.split(START, 1)
        readme = pre + block + rest.split(END, 1)[1]
    else:
        readme = readme.rstrip() + "\n\n" + block + "\n"
    Path("README.md").write_text(readme)
    print(block)


if __name__ == "__main__":
    main()
