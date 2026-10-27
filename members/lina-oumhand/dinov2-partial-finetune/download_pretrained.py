"""Download the public Meta DINOv2 ViT-S/14 checkpoint and verify its identity."""
from pathlib import Path
import hashlib
import urllib.request

URL = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth"
EXPECTED_SHA256 = "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"
path = Path(__file__).resolve().parent / "dinov2_vits14_pretrain.pth"
if not path.exists():
    partial = path.with_suffix(".part")
    with urllib.request.urlopen(URL, timeout=60) as response, partial.open("wb") as out:
        while chunk := response.read(1024 * 1024):
            out.write(chunk)
    partial.replace(path)
actual = hashlib.sha256(path.read_bytes()).hexdigest()
if actual != EXPECTED_SHA256:
    raise RuntimeError(f"Checkpoint hash mismatch: {actual}; file has not been used")
print(f"Verified {path.name}: {actual}")
