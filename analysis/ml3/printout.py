"""Ticket ML-3 P7: the D10 printout. The owner picks the seed from it.

    python analysis/ml3/printout.py > analysis/ml3/printout.txt

Runs locally from the archives only: the country runs exp261-263 (P6; predictions carry capture, photo,
folder_id and one p_<country> column per class, P1) and the live model's runs exp258-260 (true and predicted
folder id only). Nothing is rebuilt. Sections, in the plan's order (plan_country_classes.html §9):

1. Headline: per seed, val and test macro-F1 over the 8 old countries on old photos only (sessions before
   ML-3), at patch level as scored and with the argmax restricted to the 8, and at photo level (each photo's
   eval patches pooled by mean probability). Each carries a 95% CI from a bootstrap over photos.
2. The country model: patch macro-F1 over 10 countries and over the 8 old ones, all photos, per seed with
   the mean and range; per-country F1.
3. Reference (D10 b): exp258-260 remapped to countries, unpaired, under D10's caveat.
4. Per coffee (Q6 a): its val/test patches, the share predicted as its own country, its top wrong countries.
5. The ML-2 segmenter-unseen subset (Q6 b, ML-2 D21), old and new photos apart.
6. The owner's mask accept rate (D8).
7. The seed rule, with a marker on the seed it points at. Stop: the owner picks.

"Old photos" are the pools' photos outside labels/ml3/new_photos.csv; "new" are the 288 of P2 (less P5's
exclusions, which are in no split). A photo is (capture, crop name), the split key.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import f1_score

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from coffeecv.class_list import load_classes, read_coffees  # noqa: E402
from coffeecv.merge_rig import merge_cmd_args, read_exclusions  # noqa: E402

CLASSES = REPO_ROOT / "dataset" / "classes.txt"
NEW_PHOTOS = REPO_ROOT / "labels" / "ml3" / "new_photos.csv"
POOL_EXCLUDE = REPO_ROOT / "labels" / "ml3" / "pool_exclude.csv"
ML2_LISTS = REPO_ROOT / "labels" / "ml2" / "photo_lists.yaml"
EXPERIMENTS = REPO_ROOT / "experiments"
SEEDS = (42, 123, 7)
NEW_RUNS = {42: 261, 123: 262, 7: 263}
REF_RUNS = {42: 258, 123: 259, 7: 260}
# D10, computed on 2026-10-04 and quoted in the ticket: the remap must reproduce them.
REF_TICKET = {("val", 42): 0.9796, ("val", 123): 0.9826, ("val", 7): 0.9815,
              ("test", 42): 0.9785, ("test", 123): 0.9759, ("test", 7): 0.9835}
NEW_COUNTRIES = ("Peru", "Rwanda")
SPLITS = ("val", "test")
N_BOOT, BOOT_SEED = 1000, 20261006
ML2_OPUS_PASS = "97.9%"


def exp_dir(exp: int) -> Path:
    (d,) = EXPERIMENTS.glob(f"exp{exp}__*")
    return d


def stem(photo: str) -> str:
    return photo.split("__", 1)[0]


# ---------------------------------------------------------------- data

def session_pools() -> dict[str, str]:
    stages = yaml.safe_load((REPO_ROOT / "dvc.yaml").read_text())["stages"]
    out = {}
    for name, body in stages.items():
        if name.startswith("merge_segcam_"):
            pool, sessions = merge_cmd_args(body["cmd"])
            out |= {s: pool for s in sessions}
    return out


def new_photo_rows() -> list[dict]:
    """labels/ml3/new_photos.csv less P5's pool exclusions: the new photos that are in some split."""
    rows = list(csv.DictReader(l for l in NEW_PHOTOS.read_text().splitlines() if not l.startswith("#")))
    excluded = read_exclusions(POOL_EXCLUDE)
    return [r for r in rows if (r["session"], r["class_folder"], r["file"]) not in excluded]


def new_photo_keys() -> set[tuple[str, str]]:
    pools = session_pools()
    return {(pools[r["session"]], Path(r["file"]).stem) for r in new_photo_rows()}


def ml2_seen() -> set[tuple[str, str]]:
    """(pool, stem) of every photo ML-2's segmenter trained or selected on (as analysis/ml2_p6/unseen_subset)."""
    lists = yaml.safe_load(ML2_LISTS.read_text())["lists"]
    return {(Path(e["crop"]).parts[-3], stem(Path(e["crop"]).name)) for n in ("seg_train", "seg_val") for e in lists[n]}


class Split:
    """One run's archived predictions for one split."""

    def __init__(self, exp: int, split: str, keys: tuple[str, ...], new: set):
        with open(exp_dir(exp) / f"predictions_{split}.csv") as f:
            rows = list(csv.DictReader(f))
        self.keys = keys
        self.y = np.array([keys.index(r["true_label"]) for r in rows])
        self.pred = np.array([keys.index(r["pred_label"]) for r in rows])
        self.prob = np.array([[float(r[f"p_{k}"]) for k in keys] for r in rows])
        if not (self.prob.argmax(1) == self.pred).all():
            raise SystemExit(f"exp{exp} {split}: pred_label is not the argmax of the p_ columns")
        self.folder = np.array([r["folder_id"] for r in rows])
        self.photo_key = [(r["capture"], stem(r["photo"])) for r in rows]
        photos = sorted(set(self.photo_key))
        index = {p: i for i, p in enumerate(photos)}
        self.photos = photos
        self.photo = np.array([index[p] for p in self.photo_key])
        self.is_new = np.array([p in new for p in self.photo_key])
        self.photo_is_new = np.array([p in new for p in photos])


# ---------------------------------------------------------------- metrics

def macro_f1(y: np.ndarray, p: np.ndarray, k: int, labels, w: np.ndarray | None = None) -> float:
    """Macro-F1 over `labels` that occur in y (the run's macro_labels convention), from a weighted confusion."""
    cm = np.bincount(y * k + p, weights=w, minlength=k * k).reshape(k, k)
    tp = np.diag(cm)
    f1 = np.divide(2 * tp, cm.sum(0) + cm.sum(1), out=np.zeros(k), where=(cm.sum(0) + cm.sum(1)) > 0)
    present = [c for c in labels if cm[c].sum() > 0]
    return float(f1[present].mean())


def pooled(s: Split, mask: np.ndarray, cols: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per photo with patches in `mask`: (photo ids, true class, argmax over `cols` of the mean probability)."""
    ids = np.unique(s.photo[mask])
    y = np.empty(len(ids), int)
    p = np.empty(len(ids), int)
    for j, ph in enumerate(ids):
        m = mask & (s.photo == ph)
        y[j] = s.y[m][0]
        p[j] = cols[int(s.prob[m][:, cols].mean(0).argmax())]
    return ids, y, p


def boot_weights(n: int, rng: np.random.Generator) -> np.ndarray:
    """N_BOOT resamples of n photos with replacement, as per-photo multiplicities (N_BOOT x n)."""
    return np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(N_BOOT)]).astype(float)


def with_ci(score_fn, draws: np.ndarray) -> tuple[float, float, float]:
    """score_fn(weights per photo) -> F1; the point estimate and a 95% percentile CI over the draws."""
    point = score_fn(np.ones(draws.shape[1]))
    lo, hi = np.percentile([score_fn(w) for w in draws], [2.5, 97.5])
    return point, float(lo), float(hi)


def fmt_ci(t) -> str:
    return f"{t[0]:.4f} [{t[1]:.4f}, {t[2]:.4f}]"


# ---------------------------------------------------------------- sections

def headline(runs, old8):
    print("1. HEADLINE: 8 old countries, old photos only (sessions before ML-3)")
    print(f"   macro-F1 with a 95% CI (bootstrap over photos, {N_BOOT} draws, seed {BOOT_SEED}). 'as scored' takes the argmax over")
    print("   all 10 countries, so a Peru/Rwanda prediction is an error; 'restricted' takes it over the 8 old ones.")
    print("   Photo level pools each photo's eval patches by mean probability (pooled over the split's patches,")
    print("   not serving's 40). The four columns of a row share one set of resamples.\n")
    print(f"   {'seed':>4} {'split':>5} {'photos':>6} {'patches':>7}  {'patch, as scored':>25}  {'patch, restricted':>25}  "
          f"{'photo, as scored':>25}  {'photo, restricted':>25}")
    rng = np.random.default_rng(BOOT_SEED)
    k = len(runs[SEEDS[0]]["val"].keys)
    all_cols = list(range(k))
    new_pred = {}
    for seed in SEEDS:
        for split in SPLITS:
            s = runs[seed][split]
            old = ~s.is_new
            pids = np.unique(s.photo[old])
            sel = {ph: i for i, ph in enumerate(pids)}
            unit = np.array([sel.get(ph, -1) for ph in s.photo])
            restricted_pred = np.array(old8)[s.prob[:, old8].argmax(1)]

            def patch(pred):
                return lambda w: macro_f1(s.y[old], pred[old], k, old8, w[unit[old]])

            ph_ids_a, ph_y_a, ph_p_a = pooled(s, old, all_cols)
            ph_ids_r, ph_y_r, ph_p_r = pooled(s, old, old8)
            assert (ph_ids_a == pids).all() and (ph_ids_r == pids).all()
            draws = boot_weights(len(pids), rng)
            cells = [with_ci(patch(s.pred), draws), with_ci(patch(restricted_pred), draws),
                     with_ci(lambda w: macro_f1(ph_y_a, ph_p_a, k, old8, w), draws),
                     with_ci(lambda w: macro_f1(ph_y_r, ph_p_r, k, old8, w), draws)]
            new_pred[(seed, split)] = int(np.isin(s.pred[old], [s.keys.index(c) for c in NEW_COUNTRIES]).sum())
            print(f"   {seed:>4} {split:>5} {len(pids):>6} {int(old.sum()):>7}  " + "  ".join(f"{fmt_ci(c):>25}" for c in cells))
    print("\n   Old patches predicted Peru or Rwanda: " + ", ".join(f"s{sd} {sp} {n}" for (sd, sp), n in new_pred.items())
          + ". Where that is 0, 'restricted' equals 'as scored'.\n")


def country_model(runs, old8):
    print("2. THE COUNTRY MODEL: patch macro-F1, all photos (old and new)")
    keys = runs[SEEDS[0]]["val"].keys
    k = len(keys)
    table = {}
    for split in SPLITS:
        for seed in SEEDS:
            s = runs[seed][split]
            m = json.loads((exp_dir(NEW_RUNS[seed]) / "metrics.json").read_text())["splits"][split]["macro_f1"]
            f10 = macro_f1(s.y, s.pred, k, range(k))
            if abs(f10 - m) > 1e-9:
                raise SystemExit(f"exp{NEW_RUNS[seed]} {split}: recomputed {f10} != metrics.json {m}")
            per = f1_score(s.y, s.pred, labels=range(k), average=None, zero_division=0)
            table[(split, seed)] = (f10, macro_f1(s.y, s.pred, k, old8), per, len(s.y), len(s.photos))
    print(f"   {'':>12} " + " ".join(f"{'s' + str(sd):>8}" for sd in SEEDS) + f" {'mean':>8} {'range':>15}")
    for split in SPLITS:
        for label, i in (("10 countries", 0), ("8 old", 1)):
            v = [table[(split, sd)][i] for sd in SEEDS]
            print(f"   {split:>4} {label:<12}" + " ".join(f"{x:>8.4f}" for x in v)
                  + f" {np.mean(v):>8.4f} {min(v):>7.4f}-{max(v):.4f}")
    print("   (patches/photos: " + ", ".join(f"{sp} s{sd} {table[(sp, sd)][3]}/{table[(sp, sd)][4]}"
                                          for sp in SPLITS for sd in SEEDS) + ")\n")
    print("   per-country F1")
    print(f"   {'':<10} " + " ".join(f"{sp + ' s' + str(sd):>10}" for sp in SPLITS for sd in SEEDS))
    for c, key in enumerate(keys):
        print(f"   {key:<10} " + " ".join(f"{table[(sp, sd)][2][c]:>10.4f}" for sp in SPLITS for sd in SEEDS))
    print()
    print("   Note: Peru and Rwanda are one coffee each, and each camera's block of them was shot on one day:")
    blocks = defaultdict(list)
    for r in new_photo_rows():
        if r["class_folder"].split("__")[1].split("_")[0] in NEW_COUNTRIES:
            t = re.search(r"_(\d{8})_(\d{6})", r["file"])
            blocks[(r["class_folder"].split("__")[1].split("_")[0], r["session"])].append(t.group(1) + t.group(2))
    for (country, session), ts in sorted(blocks.items()):
        mins = [int(x[8:10]) * 60 + int(x[10:12]) + int(x[12:14]) / 60 for x in ts]
        print(f"     {country:<7} {session:<20} {len(ts):>3} frames in {max(mins) - min(mins):>4.0f} min (file-name times)")
    print("   and the split spreads each block across train, val and test, so near-duplicate frames sit on both")
    print("   sides of it. Their F1 is near 1.0 by construction;")
    print("   it lifts the 10-country macro-F1 and is not evidence of generalization.\n")


def reference(ref_keys):
    coffees = load_classes(CLASSES)
    print("3. REFERENCE (D10 b): the live model's runs exp258-260, remapped to countries")
    print("   Each archived patch's true and predicted folder id mapped to its country by dataset/classes.txt,")
    print("   macro-F1 recomputed over the 8 countries.\n")
    print(f"   {'seed':>4} {'split':>5} {'patches':>7} {'folder F1':>9} {'country F1':>10} {'ticket':>7} {'errors':>6} {'in-country':>10}")
    for seed in SEEDS:
        for split in SPLITS:
            with open(exp_dir(REF_RUNS[seed]) / f"predictions_{split}.csv") as f:
                rows = list(csv.DictReader(f))
            t = np.array([r["true_label"] for r in rows])
            p = np.array([r["pred_label"] for r in rows])
            ff1 = f1_score(t, p, labels=sorted(set(t)), average="macro", zero_division=0)
            tc = np.array([coffees.key_of(x) for x in t])
            pc = np.array([coffees.key_of(x) for x in p])
            cf1 = f1_score(tc, pc, labels=sorted(set(tc)), average="macro", zero_division=0)
            if round(cf1, 4) != REF_TICKET[(split, seed)]:
                raise SystemExit(f"exp{REF_RUNS[seed]} {split}: remapped {cf1:.4f}, the ticket says {REF_TICKET[(split, seed)]}")
            err = t != p
            inside = int((err & (tc == pc)).sum())
            print(f"   {seed:>4} {split:>5} {len(t):>7} {ff1:>9.4f} {cf1:>10.4f} {REF_TICKET[(split, seed)]:>7.4f} "
                  f"{int(err.sum()):>6} {inside / err.sum():>9.0%}")
    print()
    print("   D10's caveat: this reference is NOT PAIRED. It is on the old split's val/test photos, which differ from")
    print("   the new split (D7) and hold no new photos; it covers 8 classes, not 10; it maps the top-1 class only,")
    print("   since that archive keeps no probabilities. Both numbers are in-distribution near ceiling: a gap of")
    print("   about 0.01 says little. It also favours the old model: it had no Peru or Rwanda to confuse with;")
    print("   Indonesia, the one roasted class, now has two roasted neighbours (random_date_raccoon); Ethiopia and")
    print("   Brazil had two coffees' val/test support and training budget (exp258 test: 280 and 320 patches,")
    print("   against at most 160 per country now); and the new 8-country sets include the roasted")
    print("   random_date_raccoon photos. 'in-country' is the share of the old errors that were confusions inside")
    print("   one country, which the remap no longer counts.\n")


def per_coffee(runs):
    coffees = {c.folder_id: c for c in read_coffees(CLASSES)}
    print("4. PER COFFEE (Q6 a): its val/test patches, the share predicted as its own country, top wrong countries")
    print("   'not in this split': D7 split by country, and this coffee drew no photo in it.\n")
    for seed in SEEDS:
        print(f"   seed {seed} (exp{NEW_RUNS[seed]})")
        for fid, c in coffees.items():
            cells = []
            for split in SPLITS:
                s = runs[seed][split]
                m = s.folder == fid
                if not m.any():
                    cells.append(f"{split}: not in this split")
                    continue
                own = s.keys.index(c.country)
                wrong = Counter(s.keys[x] for x in s.pred[m] if x != own).most_common(2)
                cells.append(f"{split}: {int(m.sum()):>3} patches/{len(set(s.photo[m])):>2} photos, "
                             f"{(s.pred[m] == own).mean():>5.1%} own"
                             + (" (" + ", ".join(f"{w} {n}" for w, n in wrong) + ")" if wrong else ""))
            print(f"     {fid} {c.label.rstrip(';'):<40} " + "  |  ".join(f"{x:<62}" for x in cells).rstrip())
        print()


def ml2_subset(runs):
    seen = ml2_seen()
    print("5. THE ML-2 SEGMENTER-UNSEEN SUBSET (Q6 b, ML-2 D21)")
    print("   Each seed's val/test photos in neither seg_train nor seg_val; every new photo is unseen. Patch")
    print("   macro-F1 over the countries present, old and new photos apart.\n")
    print("   The new photos' countries are mostly Peru and Rwanda plus a few patches of three new coffees of old")
    print("   countries (random_date_raccoon: Sidamo, SulDeMinas, Excelso), so their macro-F1 swings on a handful of")
    print("   patches; accuracy is beside it, and section 4 has the coffees.\n")
    print(f"   {'seed':>4} {'split':>5}  {'subset':<10} {'photos':>6} {'patches':>7} {'classes':>7} {'macro-F1':>8} {'accuracy':>8}")
    for seed in SEEDS:
        for split in SPLITS:
            s = runs[seed][split]
            k = len(s.keys)
            unseen = np.array([p not in seen for p in s.photo_key])
            if (s.is_new & ~unseen).any():
                raise SystemExit("a new photo is in ML-2's segmenter lists")
            for name, m in (("full", np.ones_like(unseen)), ("old unseen", unseen & ~s.is_new), ("new", s.is_new)):
                print(f"   {seed:>4} {split:>5}  {name:<10} {len(set(s.photo[m])):>6} {int(m.sum()):>7} "
                      f"{len(set(s.y[m])):>7} {macro_f1(s.y[m], s.pred[m], k, range(k)):>8.4f} {(s.y[m] == s.pred[m]).mean():>8.4f}")
    print()


def accept_rate():
    spec = importlib.util.spec_from_file_location("mask_review", REPO_ROOT / "analysis" / "ml3" / "mask_review.py")
    mr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mr)
    from coffeecv.review_masks import current_decisions
    its = mr.read_items()
    dec = current_decisions(mr.DECISIONS, its)
    masks = [it for it in its if it["fallback"] == "0"]
    fb = [it for it in its if it["fallback"] == "1"]
    acc = sum(dec[it["item"]]["decision"] == "accept" for it in masks if it["item"] in dec)
    judged = sum(it["item"] in dec for it in masks)
    excl = Counter(r["reason"] for r in csv.DictReader(POOL_EXCLUDE.read_text().splitlines()))
    print("6. THE OWNER'S MASK ACCEPT RATE ON THE NEW PHOTOS (D8)")
    print(f"   {acc} of {judged} masks accepted ({acc / judged:.1%}); {len(its)} photos, all judged. The {len(fb)} photos")
    print(f"   the segmenter found no mask for (D18 fallback) were judged apart. By the owner's decision (P5) the")
    print(f"   pools leave out {sum(excl.values())} photos: " + ", ".join(f"{n} {r}" for r, n in excl.most_common())
          + f"; the other {len(fb) - excl['D18 fallback with background']} fallbacks")
    print("   fill the frame with beans and train whole.")
    print(f"   D8: this is the owner's rate, not comparable with ML-2's {ML2_OPUS_PASS} Opus pass rate (Opus was the")
    print("   stricter judge: the two agreed on 337 of 343 masks).\n")


def seed_rule(runs):
    val = {sd: json.loads((exp_dir(NEW_RUNS[sd]) / "metrics.json").read_text())["splits"]["val"]["macro_f1"] for sd in SEEDS}
    test = {sd: json.loads((exp_dir(NEW_RUNS[sd]) / "metrics.json").read_text())["splits"]["test"]["macro_f1"] for sd in SEEDS}
    pick = max(SEEDS, key=lambda sd: val[sd])
    print("7. THE SEED RULE")
    print("   Pick on val patch macro-F1 (10 countries, as the run scores it), as ML-1 and ML-2 did (exp260 was")
    print("   val's best), and report test. Each seed draws its own split (dataset.py:381), so comparing seeds")
    print("   partly compares val/test sets, and the picked seed's test number carries a selection bias: the card")
    print("   carries the mean and range over all three (P8).\n")
    for sd in SEEDS:
        mark = "  <-- the rule's pick" if sd == pick else ""
        print(f"   s{sd:<4} exp{NEW_RUNS[sd]}  val {val[sd]:.4f}  test {test[sd]:.4f}{mark}")
    print("\n   The OOD probe is refitted on the shipped head in P8 and checked on the spent holdout as a regression")
    print("   check only (D11): the guard is not freshly validated.")
    print("\n   STOP: the owner picks the seed.")


def main() -> int:
    classes = load_classes(CLASSES)
    sha = hashlib.sha256(CLASSES.read_bytes()).hexdigest()
    for seed in SEEDS:
        cfg = json.loads((exp_dir(NEW_RUNS[seed]) / "config.json").read_text())
        if cfg["classes_sha256"] != sha or cfg["seed"] != seed or tuple(cfg["class_ids"]) != classes.keys:
            raise SystemExit(f"exp{NEW_RUNS[seed]}: not a seed-{seed} fit on today's dataset/classes.txt")
    keys = classes.keys
    old8 = [i for i, k in enumerate(keys) if k not in NEW_COUNTRIES]
    new = new_photo_keys()
    runs = {sd: {sp: Split(NEW_RUNS[sd], sp, keys, new) for sp in SPLITS} for sd in SEEDS}
    print("ML-3 D10 printout: country model exp261-263 (seeds 42, 123, 7) on dataset/classes.txt (D15)")
    print("written by analysis/ml3/printout.py from the archived predictions\n")
    headline(runs, old8)
    country_model(runs, old8)
    reference(keys)
    per_coffee(runs)
    ml2_subset(runs)
    accept_rate()
    seed_rule(runs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
