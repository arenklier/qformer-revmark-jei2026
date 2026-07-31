"""
What happens when the location map is CORRUPTED rather than lost?

The map is read positionally: flag k tells the extractor whether candidate k
carries a bit. A single flipped flag therefore shifts the payload alignment for
every candidate after it, so damage should cascade to the end of the stream.

We measure that cascade, then test a block-synchronised map format that bounds it:
the map is cut into fixed-size blocks, each carrying its own payload-offset header,
so a corrupted block cannot desynchronise the blocks that follow.

Reported: payload bit-match rate versus map bit-error rate, plain vs blocked.
"""
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/ayhan/qformer-revmark")
sys.path.insert(0, str(ROOT / "scripts"))

from verify_revision_experiments import (  # noqa: E402
    KODAK_DIR, load_gray_uint8, embed_noshift_gray, checker_masks, rhombus_predict,
)

torch.manual_seed(0)
rng = np.random.default_rng(0)


def extract_with_map(marked, side, loc_override=None, block=0):
    """Extract using a (possibly corrupted) location map.

    block = 0  -> plain positional map (a flip desynchronises the remainder)
    block > 0  -> block-synchronised map: the payload cursor is reset at the start
                  of every block from that block's header, so damage stays local.
    """
    H, W = marked.shape
    cross, dot = checker_masks(H, W)
    T_sel = side["T"]
    loc = side["loc"] if loc_override is None else loc_override
    ncand_dot, ncand_cross = side["ncand"]

    current = marked.clone()
    payload_pos = {}          # payload index -> recovered bit

    # The payload index of candidate gi is the number of embedded flags before it.
    # Computing it from a prefix sum makes the result independent of the order in
    # which the two passes are visited.
    prefix = np.concatenate([[0], np.cumsum(np.asarray(loc, dtype=np.int64))])

    # Block headers carry the true payload offset at each block start, taken from
    # the uncorrupted map, which is what a real header would record.
    orig = side["loc"]
    if block > 0:
        oprefix = np.concatenate([[0], np.cumsum(np.asarray(orig, dtype=np.int64))])
        headers = {b: int(oprefix[min(b * block, len(orig))])
                   for b in range((len(orig) + block - 1) // block)}

    def payload_index(gi):
        if block <= 0:
            return int(prefix[gi])
        b = gi // block
        start = b * block
        return headers.get(b, 0) + int(prefix[gi] - prefix[start])

    # Extraction undoes the passes in reverse order (cross was embedded last) and
    # must restore pixels as it goes, or the second pass sees a drifted candidate set.
    for pass_mask, ncand, base in ((cross, ncand_cross, ncand_dot), (dot, ncand_dot, 0)):
        pred = rhombus_predict(current, pass_mask)
        e = current - pred
        interior = torch.zeros_like(pass_mask)
        interior[1:H - 1, 1:W - 1] = True
        cand = pass_mask & interior & (e.abs() <= T_sel + 1)
        positions = cand.flatten().nonzero(as_tuple=False).flatten().tolist()
        if len(positions) != ncand:
            raise RuntimeError(f"candidate drift: expected {ncand}, saw {len(positions)}")
        restored = current.clone()
        for k, pos in enumerate(positions):
            gi = base + k
            if gi < len(loc) and loc[gi] == 1:
                yi, xi = pos // W, pos % W
                b = int(e[yi, xi].item()) % 2
                payload_pos[payload_index(gi)] = b
                restored[yi, xi] = current[yi, xi] - b
        current = restored
    return payload_pos


def bit_match(recovered, truth):
    if not recovered:
        return 0.0
    hits = sum(1 for i, b in recovered.items() if i < len(truth) and b == truth[i])
    return hits / min(len(truth), max(recovered) + 1)


def main():
    paths = sorted(KODAK_DIR.glob("kodim*.png"))[:24]
    payload_bits = 2000
    size = 256
    bers = (0.0, 1e-5, 1e-4, 1e-3, 1e-2)
    draws = 3          # independent corruption draws per image and BER

    print(f"Kodak: {len(paths)} images, {payload_bits}-bit payload, {size}x{size} gray, "
          f"{draws} corruption draws each")
    print()

    acc = {b: dict(p=[], bl=[], rp=0, rb=0, n=0) for b in bers}
    for p in paths:
        img = load_gray_uint8(p, size)
        payload = torch.randint(0, 2, (payload_bits,), dtype=torch.int64)
        marked, side, n_emb = embed_noshift_gray(img, payload, T_sel=0)
        truth = payload.tolist()[:n_emb]
        base_loc = side["loc"]

        for ber in bers:
            reps = 1 if ber == 0 else draws
            for _ in range(reps):
                loc = list(base_loc)
                if ber > 0:
                    nflip = max(1, int(round(len(loc) * ber)))
                    idx = rng.choice(len(loc), size=min(nflip, len(loc)), replace=False)
                    for i in idx:
                        loc[i] ^= 1
                acc[ber]["n"] += 1
                # A corrupted map makes the extractor restore the wrong pixels, so the
                # second pass can see a drifted candidate set. That is a refusal, the
                # same outcome the deployed extractor produces under tampering.
                try:
                    acc[ber]["p"].append(bit_match(extract_with_map(marked, side, loc, 0), truth))
                except RuntimeError:
                    acc[ber]["rp"] += 1
                try:
                    acc[ber]["bl"].append(bit_match(extract_with_map(marked, side, loc, 256), truth))
                except RuntimeError:
                    acc[ber]["rb"] += 1

    print(f"{'map BER':>9} | {'plain: match':>13} {'refused':>9} | "
          f"{'blocked: match':>15} {'refused':>9}")
    print("-" * 64)
    for ber in bers:
        a = acc[ber]
        mp = np.mean(a["p"]) if a["p"] else float("nan")
        mb = np.mean(a["bl"]) if a["bl"] else float("nan")
        print(f"{ber:>9.0e} | {mp:>13.4f} {a['rp']:>4}/{a['n']:<4} | "
              f"{mb:>15.4f} {a['rb']:>4}/{a['n']:<4}")

    print()
    print("Reading: 1.0 = every recovered bit correct, 0.5 = chance.")
    print("A plain positional map loses alignment after the first flipped flag;")
    print("a block-synchronised map re-anchors the cursor at each block header.")


if __name__ == "__main__":
    main()
