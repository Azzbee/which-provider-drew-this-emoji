"""File-forensics features: what a PNG reveals before (and without) looking at the drawing.

Reads the raw bytes of each PNG: header, chunk list and order, IDAT segmentation, zlib settings,
per-row filter types chosen by the encoder, palette / transparency chunks, and the colour stored
under fully transparent pixels. No visible pixel content is used.
"""
import struct, zlib, hashlib, sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

KNOWN = ["IHDR","PLTE","IDAT","IEND","tRNS","gAMA","cHRM","sRGB","iCCP","sBIT","pHYs","tEXt","iTXt","zTXt",
         "tIME","bKGD","hIST","sPLT","eXIf","iDOT","CgBI","acTL","fcTL","fdAT"]
CH = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}   # channels per colour type

def chunks(b):
    i = 8
    while i + 8 <= len(b):
        n = struct.unpack(">I", b[i:i+4])[0]; t = b[i+4:i+8].decode("latin1"); d = b[i+8:i+8+n]
        yield t, d
        i += 12 + n

def features(path):
    b = Path(path).read_bytes()
    f = {"file_bytes": len(b)}
    types = []; idat = []; text_keys = []
    for t, d in chunks(b):
        types.append(t)
        if t == "IHDR":
            w, h, bd, ct, cm, fm, il = struct.unpack(">IIBBBBB", d[:13])
            f.update(width=w, height=h, bit_depth=bd, colour_type=ct, interlace=il)
        elif t == "IDAT": idat.append(d)
        elif t == "PLTE": f["palette_size"] = len(d) // 3
        elif t == "tRNS": f["trns_len"] = len(d)
        elif t == "pHYs" and len(d) >= 9:
            x, y, u = struct.unpack(">IIB", d[:9]); f["phys_x"] = x; f["phys_y"] = y; f["phys_unit"] = u
        elif t == "gAMA" and len(d) == 4: f["gamma"] = struct.unpack(">I", d)[0]
        elif t == "sRGB" and len(d) == 1: f["srgb_intent"] = d[0]
        elif t in ("tEXt", "iTXt", "zTXt"): text_keys.append(d.split(b"\0")[0].decode("latin1", "ignore"))
    for k in KNOWN: f[f"has_{k}"] = int(k in types)
    f["n_chunks"] = len(types); f["n_idat"] = len(idat)
    f["chunk_order_hash"] = int(hashlib.md5("|".join(t for t in types if t != "IDAT").encode()).hexdigest()[:8], 16)
    f["text_keys_hash"] = int(hashlib.md5("|".join(sorted(text_keys)).encode()).hexdigest()[:8], 16) if text_keys else 0
    f["n_text"] = len(text_keys)
    if idat:
        sizes = [len(d) for d in idat]
        f["idat_first"] = sizes[0]; f["idat_max"] = max(sizes); f["idat_total"] = sum(sizes)
        z = b"".join(idat)
        f["zlib_cmf"] = z[0]; f["zlib_flg"] = z[1]; f["zlib_level"] = z[1] >> 6
        # deflate block type of the first block (stored / fixed / dynamic Huffman)
        f["deflate_btype"] = (z[2] >> 1) & 3 if len(z) > 2 else -1
        try:
            raw = zlib.decompress(z)
            ch = CH.get(f.get("colour_type", 6), 4); bpp = max(1, ch * f.get("bit_depth", 8) // 8)
            stride = 1 + (f["width"] * ch * f.get("bit_depth", 8) + 7) // 8
            if f.get("interlace", 0) == 0 and len(raw) >= stride * f["height"]:
                filt = np.frombuffer(raw, np.uint8)[: stride * f["height"]].reshape(f["height"], stride)[:, 0]
                for k in range(5): f[f"filter_{k}"] = float((filt == k).mean())
                f["filter_first"] = int(filt[0]); f["filter_nunique"] = int(len(set(filt.tolist())))
            f["compress_ratio"] = len(z) / max(1, len(raw))
            # how well a standard zlib at level 9 would do: encoder efficiency fingerprint
            f["recompress_gain"] = len(z) / max(1, len(zlib.compress(raw, 9)))
        except Exception:
            f["compress_ratio"] = -1
    f["bytes_per_pixel"] = len(b) / max(1, f.get("width", 1) * f.get("height", 1))
    # colour stored under fully transparent pixels (invisible on screen)
    try:
        im = Image.open(path)
        a = np.asarray(im.convert("RGBA"))
        tr = a[..., 3] == 0
        f["frac_transparent"] = float(tr.mean())
        if tr.any():
            u = a[..., :3][tr].astype(float)
            f["under_r"], f["under_g"], f["under_b"] = u.mean(0); f["under_std"] = float(u.std())
            f["under_n_unique"] = int(len(np.unique(u[:2000], axis=0)))
        f["alpha_n_unique"] = int(len(np.unique(a[..., 3])))
    except Exception:
        pass
    return f

if __name__ == "__main__":
    root = Path(sys.argv[1]); out = Path(sys.argv[2])
    rows = []
    for split in ("train", "test"):
        for p in sorted((root / split).glob("*.png")):
            r = features(p); r["Id"] = p.stem; r["split"] = split; rows.append(r)
    pd.DataFrame(rows).to_csv(out, index=False); print("wrote", out, len(rows))
