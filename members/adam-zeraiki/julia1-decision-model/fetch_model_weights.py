#!/usr/bin/env python3
"""Fetch the public, pinned Julia-1 weights and install only after verification.

Uses only the Python standard library. No account or credential is required.
The tokenizer and model source are already present in the release archive.
"""
from __future__ import annotations
import argparse
import hashlib
import os
from pathlib import Path
import tempfile
import urllib.request

REVISION = 'a85b127321d580d65176c89ced8273f305745d85'
URL = f'https://huggingface.co/SupersonicLabs/Julia-1/resolve/{REVISION}/model.safetensors'
EXPECTED_BYTES = 577189056
EXPECTED_SHA256 = 'df853bf7fe424420011f3d0c47a05d7341aa9eefa7fb9f203ea4aada4ad95b72'
DEFAULT_PATH = Path(__file__).resolve().parent / 'jev/julia_model/model.safetensors'


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_PATH)
    destination = parser.parse_args().output.resolve()
    if destination.exists():
        if destination.is_file() and destination.stat().st_size == EXPECTED_BYTES and digest(destination) == EXPECTED_SHA256:
            print(f'Weights already verified: {destination}')
            return
        raise SystemExit(f'Existing target differs from the pinned weights; move it aside first: {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=destination.name + '.', suffix='.partial', dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        received = 0
        value = hashlib.sha256()
        request = urllib.request.Request(URL, headers={'User-Agent': 'Julia1-Emoji-Decision-Replay/1.0'})
        print(f'Downloading {EXPECTED_BYTES:,} bytes from {URL}', flush=True)
        with os.fdopen(handle, 'wb') as target, urllib.request.urlopen(request, timeout=60) as response:
            if response.status != 200:
                raise RuntimeError(f'Unexpected HTTP status: {response.status}')
            while chunk := response.read(4 * 1024 * 1024):
                received += len(chunk)
                if received > EXPECTED_BYTES:
                    raise RuntimeError('Download exceeds the pinned checkpoint size')
                value.update(chunk)
                target.write(chunk)
            target.flush()
            os.fsync(target.fileno())
        if received != EXPECTED_BYTES or value.hexdigest() != EXPECTED_SHA256:
            raise RuntimeError(f'Checkpoint verification failed: bytes={received}, sha256={value.hexdigest()}')
        if destination.exists():
            raise RuntimeError('Target appeared during the download; existing file was left unchanged')
        os.replace(temporary, destination)
        print(f'Installed verified weights: {destination}\nSHA256: {EXPECTED_SHA256}')
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
