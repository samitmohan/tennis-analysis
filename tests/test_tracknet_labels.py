"""Label.csv parsing: an event labelled on consecutive frames is one event."""

from evaluation.tracknet import load_clips


def test_consecutive_event_frames_merge(tmp_path):
    clip = tmp_path / "game1" / "Clip1"
    clip.mkdir(parents=True)
    status = [0, 1, 1, 0, 2, 0, 1, 1, 1, 2, 2, 0, 1]
    rows = ["file name,visibility,x-coordinate,y-coordinate,status"]
    rows += [f"{i:04d}.jpg,1,{100 + i},{200 + i},{s}" for i, s in enumerate(status)]
    (clip / "Label.csv").write_text("\n".join(rows) + "\n")
    (c,) = load_clips(tmp_path, ["game1"])
    assert c.hits == [1, 6, 12]
    assert c.bounces == [4, 9]
