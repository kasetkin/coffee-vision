"""ML-7 P7: how far each model's values reach, for fp16 (plan P7). Every photo of a list (p7_range_photos.txt:
every photo under dataset/, the true negatives included) runs serving's own path in PyTorch fp32: the segmenter
inside patches_for_photo, then DINOv3 on the 40 patches it cuts. Recorded per photo, per site:

  out:<module>        the largest |value| of each module's output (every module of the encoder, the prompt
                      encoder, the mask decoder and DINOv3, the residual blocks included)
  lmla:<LiteMLA>.*    inside each linear attention: q, k (after ReLU) and v; vk_v = v k^T and ksum = 1 k^T, sums
                      over the 4,096 tokens; num = vk_v q and den = ksum q; den's least value above 0, and how
                      many dens are exactly 0 (den_zero: 0/(0 + eps) in fp32, 0/0 once eps underflows); out
  ln:<LayerNorm>.*    DINOv3's LayerNorm inputs: |x|, Σx² over the channels, the least variance
  attn:<Attention>    DINOv3's attention logits, (q · scale) k^T

fp16's largest finite value is 65504 and its least normal one 6.1e-5; the summary gives each site's extreme over
the photos, which photo and role reached it, and the headroom in powers of two. Training environment, from the
repo root (resumable: photos already in OUT are skipped; --shard k/n splits the list between processes):

    python analysis/ml7_gpu/p7_ranges.py scan --checkpoint models/allrigs_dino3b16_segxl0v3_country_s42.pt \\
        --photos analysis/ml7_gpu/p7_range_photos.txt --out analysis/ml7_gpu/scratch/p7_ranges --threads 8
    python analysis/ml7_gpu/p7_ranges.py summary --out analysis/ml7_gpu/scratch/p7_ranges \\
        --json analysis/ml7_gpu/p7_ranges.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import types
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import graphs  # noqa: E402

FP16_MAX, FP16_MIN_NORMAL = 65504.0, 6.103515625e-05
LEAST = (".den_min", ".var_min")          # sites whose least value is the one that matters


class Ranges:
    """The current photo's extremes, per site."""

    def __init__(self):
        self.v: dict[str, float] = {}

    def most(self, site: str, x: torch.Tensor) -> None:
        if x.numel():
            self.v[site] = max(self.v.get(site, 0.0), float(x.detach().abs().max()))

    def count(self, site: str, n: int) -> None:
        self.v[site] = max(self.v.get(site, 0), int(n))

    def least(self, site: str, x: torch.Tensor) -> None:
        if x.numel():
            self.v[site] = min(self.v.get(site, math.inf), float(x.detach().min()))


def _liteMLA_measured(ranges: Ranges, name: str):
    def relu_linear_att(self, qkv: torch.Tensor) -> torch.Tensor:
        """efficientvit's LiteMLA.relu_linear_att (fp32), measured on the way."""
        B, _, H, W = list(qkv.size())
        qkv = torch.reshape(qkv, (B, -1, 3 * self.dim, H * W))
        q, k, v = qkv[:, :, 0:self.dim], qkv[:, :, self.dim:2 * self.dim], qkv[:, :, 2 * self.dim:]
        q, k = self.kernel_func(q), self.kernel_func(k)
        v = torch.nn.functional.pad(v, (0, 0, 0, 1), mode="constant", value=1)
        vk = torch.matmul(v, k.transpose(-1, -2))
        out = torch.matmul(vk, q)
        num, den = out[:, :, :-1], out[:, :, -1:]
        for site, x in (("q", q), ("k", k), ("v", v[:, :, :-1]), ("vk_v", vk[:, :, :-1]), ("ksum", vk[:, :, -1:]),
                        ("num", num), ("den", den)):
            ranges.most(f"lmla:{name}.{site}", x)
        ranges.least(f"lmla:{name}.den_min", den.masked_fill(den == 0, math.inf))   # the least one above 0
        ranges.count(f"lmla:{name}.den_zero", (den == 0).sum())
        out = num / (den + self.eps)
        ranges.most(f"lmla:{name}.out", out)
        return torch.reshape(out, (B, -1, H, W))
    return relu_linear_att


def _attention_measured(ranges: Ranges, name: str):
    def forward(self, x, rope=None, attn_mask=None, is_causal=False):
        """graphs._attention_4d (timm's EvaAttention written out), its logits measured."""
        if attn_mask is not None or is_causal:
            raise ValueError("no attention mask expected")
        B, N, _ = x.shape
        q, k, v = (t.reshape(B, N, self.num_heads, -1).transpose(1, 2) for t in self.qkv(x).chunk(3, dim=-1))
        q, k = self.q_norm(q), self.k_norm(k)
        if rope is not None:
            from timm.layers import apply_rot_embed_cat
            npt, half = self.num_prefix_tokens, getattr(self, "rotate_half", False)
            q = torch.cat([q[:, :, :npt, :], apply_rot_embed_cat(q[:, :, npt:, :], rope, half=half)], dim=2).type_as(v)
            k = torch.cat([k[:, :, :npt, :], apply_rot_embed_cat(k[:, :, npt:, :], rope, half=half)], dim=2).type_as(v)
        logits = torch.matmul(q * self.scale, k.transpose(-2, -1))
        ranges.most(f"attn:{name}", logits)
        x = torch.matmul(logits.softmax(dim=-1), v).transpose(1, 2).reshape(B, N, -1)
        return self.proj(self.norm(x))
    return forward


def instrument(ctx, ranges: Ranges) -> int:
    """Hooks on every module of the three graphs; LiteMLA and DINOv3's attention measured inside. Sites hooked."""
    from efficientvit.models.nn.ops import LiteMLA
    from timm.models.eva import EvaAttention

    sam = ctx.seg.model
    roots = {"enc": sam.image_encoder, "prompt": sam.prompt_encoder, "dec": sam.mask_decoder, "dino": ctx.model}
    n = 0
    for tag, root in roots.items():
        for name, m in root.named_modules():
            full = f"{tag}.{name}" if name else tag

            def hook(_m, _inp, out, site=f"out:{full}"):
                for t in out if isinstance(out, (tuple, list)) else (out,):
                    if isinstance(t, torch.Tensor) and t.is_floating_point():
                        ranges.most(site, t)
            m.register_forward_hook(hook)
            n += 1
            if isinstance(m, LiteMLA):
                m.relu_linear_att = types.MethodType(_liteMLA_measured(ranges, full), m)
            elif isinstance(m, EvaAttention):
                m.forward = types.MethodType(_attention_measured(ranges, full), m)
            elif tag == "dino" and isinstance(m, torch.nn.LayerNorm):
                def pre(_m, inp, site=f"ln:{full}"):
                    x = inp[0]
                    ranges.most(f"{site}.x", x)
                    ranges.most(f"{site}.sumsq", (x * x).sum(-1))
                    ranges.least(f"{site}.var_min", x.var(-1, unbiased=False))
                m.register_forward_pre_hook(pre)
    return n


def read_photos(path: Path) -> list[tuple[str, str]]:
    """(role, path under the repo) per line of a `<role> <sha256> <path>` list."""
    out = []
    for line in path.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            role, _, rel = line.split(maxsplit=2)
            out.append((role, rel))
    return out


@torch.no_grad()
def scan(args) -> None:
    from coffeecv.infer import forward_with_embeddings, patches_for_photo
    from coffeecv.transforms import build_eval_transform
    from make_refs import Context

    torch.set_num_threads(args.threads)
    k, n = (int(x) for x in args.shard.split("/"))
    photos = read_photos(args.photos)[k - 1::n]
    args.out.mkdir(parents=True, exist_ok=True)
    rows = args.out / f"shard_{k}of{n}.jsonl"
    done = {json.loads(line)["path"] for line in rows.read_text().splitlines()} if rows.exists() else set()
    ctx = Context.load(args.checkpoint)
    transform = build_eval_transform(ctx.cfg.patch_resize)
    ranges = Ranges()
    print(f"hooked {instrument(ctx, ranges)} modules; {len(photos)} photos, {len(done)} done", flush=True)
    t0 = time.perf_counter()
    with rows.open("a") as f:
        for i, (role, rel) in enumerate(photos):
            if rel in done:
                continue
            ranges.v, t = {}, time.perf_counter()
            row = {"path": rel, "role": role}
            try:
                patches, info = patches_for_photo(graphs.REPO / rel, ctx.cfg, 40, [42, 0])
                forward_with_embeddings(ctx.model, ctx.head_module, torch.stack([transform(p) for p in patches]))
                row |= {"fallback": bool(info.get("seg_fallback")), "mask_area_frac": info.get("mask_area_frac"),
                        "v": ranges.v}
            except Exception as e:                                      # noqa: BLE001  kept by name, the scan goes on
                row["error"] = f"{type(e).__name__}: {e}"
            row["s"] = round(time.perf_counter() - t, 2)
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"photo {i + 1}/{len(photos)} {role} {Path(rel).name} {row['s']:.1f} s"
                  f"{' ERROR ' + row['error'] if 'error' in row else ''} (elapsed {time.perf_counter() - t0:.0f} s)",
                  flush=True)


def summary(args) -> None:
    rows = [json.loads(line) for f in sorted(args.out.glob("shard_*.jsonl")) for line in f.read_text().splitlines()]
    ok = [r for r in rows if "v" in r]
    sites: dict[str, dict] = {}
    for r in ok:
        for site, x in r["v"].items():
            s = sites.setdefault(site, {"by_role": {}})
            least = site.endswith(LEAST)
            better = (lambda a, b: a < b) if least else (lambda a, b: a > b)
            if "value" not in s or better(x, s["value"]):
                s.update(value=x, photo=r["path"], role=r["role"])
            br = s["by_role"]
            if r["role"] not in br or better(x, br[r["role"]]):
                br[r["role"]] = x
    for site, s in sites.items():
        x = s["value"]
        if site.endswith(".den_zero"):
            s["headroom_log2"] = None
            continue
        if site.endswith(LEAST):
            s["headroom_log2"] = round(math.log2(x / FP16_MIN_NORMAL), 2) if x > 0 else None
        else:
            s["headroom_log2"] = round(math.log2(FP16_MAX / x), 2) if x > 0 else None
    tight = sorted((s["headroom_log2"], site) for site, s in sites.items() if s["headroom_log2"] is not None)
    out = {"photos": len(rows), "scanned": len(ok), "errors": {r["path"]: r["error"] for r in rows if "error" in r},
           "roles": {role: sum(r["role"] == role for r in ok) for role in sorted({r["role"] for r in ok})},
           "fallback_photos": sum(bool(r.get("fallback")) for r in ok),
           "fp16": {"max": FP16_MAX, "min_normal": FP16_MIN_NORMAL},
           "beyond_fp16": [site for h, site in tight if h < 0],
           "den_zero": {site: s["value"] for site, s in sites.items() if site.endswith(".den_zero") and s["value"]},
           "tightest": [{"site": site, **sites[site]} for _, site in tight[:60]],
           "sites": sites}
    args.json.write_text(json.dumps(out, indent=1) + "\n")
    print(f"{len(ok)} photos scanned ({len(rows) - len(ok)} errors), {len(sites)} sites, "
          f"{len(out['beyond_fp16'])} beyond fp16")
    for h, site in tight[:25]:
        s = sites[site]
        print(f"  {h:7.2f}  {site:70s} {s['value']:.4g}  {s['role']} {Path(s['photo']).name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan")
    s.add_argument("--checkpoint", type=Path, required=True)
    s.add_argument("--photos", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--threads", type=int, default=8)
    s.add_argument("--shard", default="1/1", help="k/n: every n-th photo from the k-th")
    m = sub.add_parser("summary")
    m.add_argument("--out", type=Path, required=True)
    m.add_argument("--json", type=Path, required=True)
    args = ap.parse_args()
    scan(args) if args.cmd == "scan" else summary(args)


if __name__ == "__main__":
    main()
