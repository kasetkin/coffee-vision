"""The segmenter dataset builder (ticket ML-5 P5; D11-D13, D16). Plain unittest.

    python -m unittest discover -s tests -p 'test_seg_dataset.py'

`assign` is tested on synthetic photo records; the committed labels/ml5/seg_dataset.yaml is checked as it is
(git-tracked only), and against the photos when they are checked out (@real_data).
"""
from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import yaml

from coffeecv import seg_dataset as sd
from coffeecv.config import REPO_ROOT, RunConfig

from tests._tiers import real_data

PARAMS = sd.Params(seed=5, split={"train": 0.60, "validation": 0.15, "test": 0.25})
T0 = datetime(2026, 9, 11, 13, 0, 0)


def photo(path: str, source: str, sha: str | None = None, **kw) -> dict:
    """A candidate record as `scan` returns it; the sha256 defaults to one unique to the path."""
    rec = {"path": path, "sha256": sha or hashlib.sha256(path.encode()).hexdigest(), "source": source, "time": None,
           "url": ""}
    if source == "pool":
        rec["folder"] = path.split("/")[1]
    if source == "negative":
        rec.setdefault("batch", "b1")
        rec.setdefault("tag", "t1")
    rec.update(kw)
    return rec


def positives(n: int, prefix: str = "sp") -> list[dict]:
    return [photo(f"dataset/segmenter_positives/{prefix}{i:03d}.jpg", "segmenter_positive") for i in range(n)]


def pool(folder: str, n: int) -> list[dict]:
    return [photo(f"dataset/{folder}/class_001__X/p{i:03d}.jpg", "pool") for i in range(n)]


def by_source(entries: list[dict]) -> Counter:
    return Counter(e["source"] for e in entries)


class PoolDraw(unittest.TestCase):
    def test_draws_as_many_pool_photos_as_segmenter_positives(self):
        out = sd.assign(positives(10) + pool("a", 30) + pool("b", 30), [], PARAMS)
        self.assertEqual(by_source(out), {"segmenter_positive": 10, "pool": 10})

    def test_draw_is_proportional_to_folder_size(self):
        out = sd.assign(positives(20) + pool("big", 75) + pool("small", 25), [], PARAMS)
        self.assertEqual(Counter(e["path"].split("/")[1] for e in out if e["source"] == "pool"),
                         {"big": 15, "small": 5})

    def test_skips_pool_photos_byte_identical_to_a_segmenter_positive(self):
        sp = positives(3)
        copies = [photo(f"dataset/a/class_001__X/c{i}.jpg", "pool", sha=p["sha256"]) for i, p in enumerate(sp)]
        out = sd.assign(sp + copies + pool("a", 3), [], PARAMS)
        self.assertFalse({e["path"] for e in out} & {c["path"] for c in copies})
        self.assertEqual(by_source(out)["pool"], 3)


def split_counts(entries: list[dict], **match) -> dict:
    return dict(Counter(e["split"] for e in entries if all(e.get(k) == v for k, v in match.items())))


class Split(unittest.TestCase):
    def test_sixty_fifteen_twenty_five_within_each_source(self):
        out = sd.assign(positives(100) + pool("a", 300), [], PARAMS)
        for source in ("segmenter_positive", "pool"):
            self.assertEqual(split_counts(out, source=source), {"train": 60, "validation": 15, "test": 25})

    def test_negatives_split_within_each_batch_and_tag(self):
        negs = [photo(f"dataset/ood_negatives/{b}/{t}/n{i}.jpg", "negative", batch=b, tag=t)
                for b, t in (("b1", "empty_tray"), ("b1", "ground_coffee"), ("b2", "empty_tray"))
                for i in range(20)]
        out = sd.assign(negs, [], PARAMS)
        for b, t in (("b1", "empty_tray"), ("b1", "ground_coffee"), ("b2", "empty_tray")):
            self.assertEqual(split_counts(out, batch=b, tag=t), {"train": 12, "validation": 3, "test": 5})

    def test_the_same_seed_gives_the_same_split(self):
        data = positives(40) + pool("a", 80)
        self.assertEqual(sd.assign(data, [], PARAMS), sd.assign(data, [], PARAMS))
        other = sd.assign(data, [], sd.Params(seed=6, split=PARAMS.split))
        self.assertNotEqual([e["split"] for e in sd.assign(data, [], PARAMS)], [e["split"] for e in other])


def at(seconds: float, clock: str = "utc") -> tuple:
    return (clock, T0 + timedelta(seconds=seconds))


def entry(entries: list[dict], path: str) -> dict:
    return next(e for e in entries if e["path"] == path)


class Groups(unittest.TestCase):
    def setUp(self):
        self.background = positives(40, prefix="bg") + pool("a", 200)

    def assert_one_group(self, out: list[dict], paths: list[str]):
        got = {(entry(out, p)["group"], entry(out, p)["split"]) for p in paths}
        self.assertEqual(len(got), 1, got)
        self.assertEqual(got.pop()[0], min(paths))

    def test_byte_identical_files_are_one_group(self):
        a = photo("dataset/segmenter_positives/x1.jpg", "segmenter_positive", sha="a" * 64)
        b = photo("dataset/ood_positives_internet/beans/x2.jpg", "internet_positive", sha="a" * 64)
        out = sd.assign(self.background + [a, b], [], PARAMS)
        self.assert_one_group(out, [a["path"], b["path"]])

    def test_shots_chain_while_each_is_within_120_s_of_the_last(self):
        burst = [photo(f"dataset/segmenter_positives/t{i}.jpg", "segmenter_positive", time=at(s))
                 for i, s in enumerate((0, 100, 220, 341))]
        out = sd.assign(self.background + burst, [], PARAMS)
        self.assert_one_group(out, [b["path"] for b in burst[:3]])
        self.assertNotEqual(entry(out, burst[3]["path"])["group"], entry(out, burst[0]["path"])["group"])

    def test_times_on_different_clocks_never_group(self):
        a = photo("dataset/segmenter_positives/c1.jpg", "segmenter_positive", time=at(0, "utc"))
        b = photo("dataset/segmenter_positives/c2.jpg", "segmenter_positive", time=at(0, "local OnePlus KB2005"))
        out = sd.assign(self.background + [a, b], [], PARAMS)
        self.assertNotEqual(entry(out, a["path"])["group"], entry(out, b["path"])["group"])

    def test_a_negative_shot_seconds_after_a_bean_photo_is_another_subject(self):
        sp = photo("dataset/segmenter_positives/b.jpg", "segmenter_positive", time=at(0))
        neg = photo("dataset/ood_negatives/b1/empty_tray/n.jpg", "negative", tag="empty_tray", time=at(30))
        out = sd.assign(self.background + [sp, neg], [], PARAMS)
        self.assertNotEqual(entry(out, sp["path"])["group"], entry(out, neg["path"])["group"])

    def test_negatives_of_two_tags_shot_seconds_apart_are_two_subjects(self):
        a = photo("dataset/ood_negatives/b1/t1/n.jpg", "negative", tag="confusable_grain", time=at(0))
        b = photo("dataset/ood_negatives/b1/t2/n.jpg", "negative", tag="green_legume", time=at(30))
        out = sd.assign(self.background + [a, b], [], PARAMS)
        self.assertNotEqual(entry(out, a["path"])["group"], entry(out, b["path"])["group"])

    def test_a_group_across_sources_shares_one_split(self):
        sp = [photo(f"dataset/segmenter_positives/s{i}.jpg", "segmenter_positive", time=at(1000 * i))
              for i in range(40)]
        pl = [photo(f"dataset/a/class_001__X/q{i}.jpg", "pool", time=at(1000 * i + 30)) for i in range(40)]
        out = sd.assign(sp + pl, [], PARAMS)
        for s, q in zip(sp, pl):
            if q["path"] in {e["path"] for e in out}:
                self.assert_one_group(out, [s["path"], q["path"]])

    def test_photos_from_one_source_url_are_one_group(self):
        negs = [photo(f"dataset/ood_negatives/b1/t1/u{i}.jpg", "negative", url="https://example.org/a.jpg")
                for i in range(2)]
        out = sd.assign(self.background + negs, [], PARAMS)
        self.assert_one_group(out, [n["path"] for n in negs])


def stored(entries: list[dict]) -> list[dict]:
    """Entries as the file stores them: no shot time or URL."""
    return [{k: v for k, v in e.items() if k not in ("time", "url")} for e in entries]


class AppendOnly(unittest.TestCase):
    def setUp(self):
        self.data = positives(40) + pool("a", 150) + pool("b", 50)
        self.first = stored(sd.assign(self.data, [], PARAMS))

    def test_a_rebuild_on_the_same_data_changes_nothing(self):
        self.assertEqual(stored(sd.assign(self.data, self.first, PARAMS)), self.first)

    def test_a_new_segmenter_positive_is_added_and_the_pool_tops_up_by_one(self):
        new = positives(1, prefix="new")
        out = stored(sd.assign(self.data + new, self.first, PARAMS))
        self.assertEqual([e for e in out if e["path"] in {f["path"] for f in self.first}], self.first)
        self.assertEqual(by_source(out), {"segmenter_positive": 41, "pool": 41})
        self.assertIn(entry(out, new[0]["path"])["split"], sd.SPLITS)

    def test_a_new_photo_joins_its_groups_split_and_keeps_the_groups_name(self):
        data = [dict(p, time=at(1000 * i)) for i, p in enumerate(positives(40))] + pool("a", 100)
        first = stored(sd.assign(data, [], PARAMS))
        old = data[7]
        new = photo("dataset/segmenter_positives/a_new.jpg", "segmenter_positive", time=at(7000 + 20))
        out = sd.assign(data + [new], first, PARAMS)
        self.assertEqual((entry(out, new["path"])["split"], entry(out, new["path"])["group"]),
                         (entry(first, old["path"])["split"], entry(first, old["path"])["group"]))

    def test_refuses_a_listed_photo_gone_from_the_data(self):
        with self.assertRaisesRegex(ValueError, "append-only"):
            sd.assign(self.data[1:], self.first, PARAMS)

    def test_refuses_a_listed_photo_whose_bytes_changed(self):
        changed = [dict(self.data[0], sha256="f" * 64)] + self.data[1:]
        with self.assertRaisesRegex(ValueError, "append-only"):
            sd.assign(changed, self.first, PARAMS)

    def test_refuses_a_new_photo_that_joins_groups_in_two_splits(self):
        data = [dict(p, time=at(1000 * i)) for i, p in enumerate(positives(40))] + pool("a", 100)
        first = stored(sd.assign(data, [], PARAMS))
        by_split = {}
        for p in data[:40]:
            by_split.setdefault(entry(first, p["path"])["split"], p)
        a, b = by_split["train"], by_split["test"]
        bridge = [photo(f"dataset/segmenter_positives/z{i}.jpg", "segmenter_positive", time=(a["time"][0], s))
                  for i, s in enumerate(_steps(a["time"][1], b["time"][1]))]
        with self.assertRaisesRegex(ValueError, "two splits"):
            sd.assign(data + bridge, first, PARAMS)


def _steps(t0: datetime, t1: datetime) -> list[datetime]:
    """Times every 100 s from t0 to t1, so the shots chain the two into one group."""
    lo, hi = min(t0, t1), max(t0, t1)
    out, t = [], lo + timedelta(seconds=100)
    while t < hi:
        out.append(t)
        t += timedelta(seconds=100)
    return out


class ShotTime(unittest.TestCase):
    def test_exif_time_with_its_offset_is_utc(self):
        exif = {"DateTimeOriginal": "2026:09:11 18:00:00", "OffsetTimeOriginal": "+05:00", "Model": "G8441"}
        self.assertEqual(sd.shot_time(exif, "sony", "PIC_20260911_180000.JPG"), ("utc", T0))

    def test_exif_time_without_an_offset_is_that_cameras_local_clock(self):
        exif = {"DateTimeOriginal": "2026:10:08 09:28:17", "Make": "OnePlus", "Model": "KB2005"}
        self.assertEqual(sd.shot_time(exif, "oneplus_kb2005", "IMG_20261008_092817.jpg"),
                         ("local OnePlus KB2005", datetime(2026, 10, 8, 9, 28, 17)))

    def test_without_exif_a_pixel_file_name_is_utc(self):
        self.assertEqual(sd.shot_time({}, "pixel", "PXL_20260911_130000123.jpg"), ("utc", T0))

    def test_without_exif_another_cameras_file_name_is_its_local_clock(self):
        self.assertEqual(sd.shot_time({}, "sony", "PIC_20260911_130000.JPG"), ("local sony", T0))

    def test_no_exif_time_and_no_time_in_the_name_is_none(self):
        self.assertIsNone(sd.shot_time({}, "internet", "internet_beans_001.jpg"))


class Reader(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name) / "seg_dataset.yaml"
        negs = [photo(f"dataset/ood_negatives/b1/empty_tray/n{i}.jpg", "negative", tag="empty_tray")
                for i in range(4)]
        self.entries = stored(sd.assign(positives(4) + pool("a", 8) + negs, [], PARAMS))

    def tearDown(self):
        self.dir.cleanup()

    def write_photos(self, photos: list[dict]) -> None:
        self.path.write_text(yaml.safe_dump({"meta": {}, "photos": photos}, sort_keys=False))

    def test_reads_back_what_was_written(self):
        sd.write_seg_dataset({"seed": 5}, self.entries, self.path)
        self.assertEqual(sd.load_seg_dataset(self.path), self.entries)

    def test_refuses_a_negative_without_its_tag(self):
        bad = [dict(e) for e in self.entries]
        del next(e for e in bad if e["source"] == "negative")["tag"]
        self.write_photos(bad)
        with self.assertRaisesRegex(ValueError, "missing tag"):
            sd.load_seg_dataset(self.path)

    def test_refuses_a_split_not_written_out_in_full(self):
        self.write_photos([dict(self.entries[0], split="val")] + self.entries[1:])
        with self.assertRaisesRegex(ValueError, "split 'val'"):
            sd.load_seg_dataset(self.path)

    def test_refuses_a_path_listed_twice(self):
        self.write_photos(self.entries + self.entries[:1])
        with self.assertRaisesRegex(ValueError, "listed twice"):
            sd.load_seg_dataset(self.path)


class CommittedFile(unittest.TestCase):
    """labels/ml5/seg_dataset.yaml as committed (git-tracked, no photos needed)."""

    @classmethod
    def setUpClass(cls):
        cls.meta, cls.photos = sd.read_seg_dataset()

    def test_no_group_in_two_splits(self):
        splits = defaultdict(set)
        for e in self.photos:
            splits[e["group"]].add(e["split"])
        self.assertEqual({g: s for g, s in splits.items() if len(s) > 1}, {})

    def test_no_test_sha256_in_training_or_validation(self):
        test = {e["sha256"] for e in self.photos if e["split"] == "test"}
        self.assertEqual([e["path"] for e in self.photos if e["split"] != "test" and e["sha256"] in test], [])

    def test_byte_identical_photos_share_a_group(self):
        groups = defaultdict(set)
        for e in self.photos:
            groups[e["sha256"]].add(e["group"])
        self.assertEqual({s: g for s, g in groups.items() if len(g) > 1}, {})

    def test_every_source_in_full_and_as_many_pool_photos_as_segmenter_positives(self):
        # Ticket ML-5 §2 and D3/D12/D16: 128 segmenter positives, 142 negatives in three batches, 12 internet.
        self.assertEqual(by_source(self.photos),
                         {"segmenter_positive": 128, "pool": 128, "negative": 142, "internet_positive": 12})

    def test_no_pool_photo_is_a_copy_of_a_segmenter_positive(self):
        positive = {e["sha256"] for e in self.photos if e["source"] == "segmenter_positive"}
        self.assertEqual([e["path"] for e in self.photos if e["source"] == "pool" and e["sha256"] in positive], [])

    def test_built_with_params_yaml_seed_and_shares(self):
        block = RunConfig.from_params_yaml().seg_dataset
        self.assertEqual((self.meta["seed"], self.meta["split"]), (block["seed"], block["split"]))

    def test_counts_table_totals_every_photo(self):
        table = sd.counts_table(self.photos)
        print(f"\n{table}", file=sys.stderr)
        self.assertEqual(table.splitlines()[-1].split()[-1], str(len(self.photos)))


@real_data
class CommittedFileAgainstPhotos(unittest.TestCase):
    """The file against the checked-out photos (`dvc checkout dataset/*.dvc dataset/ood_negatives/*.dvc`)."""

    @classmethod
    def setUpClass(cls):
        cls.photos = sd.load_seg_dataset()
        cls.candidates = sd.scan()

    def test_every_photo_exists_with_its_bytes(self):
        bad = [e["path"] for e in self.photos if not (REPO_ROOT / e["path"]).is_file()
               or sd.sha256_file(REPO_ROOT / e["path"]) != e["sha256"]]
        self.assertEqual(bad, [])

    def test_a_rebuild_from_the_data_changes_nothing(self):
        params = sd.Params.from_config(RunConfig.from_params_yaml().seg_dataset)
        self.assertEqual(stored(sd.assign(self.candidates, self.photos, params)), self.photos)


if __name__ == "__main__":
    unittest.main()
