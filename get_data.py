#!/usr/bin/env python3
"""Download, verify and unpack the numerical data for the paper."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parent


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("replay", "inference"), default="replay")
    parser.add_argument("--archive", type=Path, help="Use an already downloaded archive")
    args = parser.parse_args()
    metadata = json.loads((ROOT / "data/releases.json").read_text())[args.kind]
    path = args.archive.resolve() if args.archive else ROOT / ".downloads" / metadata["filename"]
    if not path.is_file():
        if args.archive:
            raise FileNotFoundError(path)
        if not metadata.get("url") and not metadata.get("parts"):
            parser.error("The large inference archive is not attached to this release. Supply --archive for a separately obtained copy, or regenerate input bundles with the production scripts; see README.md.")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".download")
        print(f"Downloading {metadata['filename']} ({metadata['bytes'] / 1e6:.1f} MB)", flush=True)
        parts = metadata.get("parts", [metadata])
        with temporary.open("wb") as target:
            for index, part in enumerate(parts):
                print(f"Downloading part {index + 1}/{len(parts)}", flush=True)
                request = urllib.request.Request(part["url"], headers={"User-Agent": "taiji-sgwb-paper-reproduction"})
                digest = hashlib.sha256()
                size = 0
                with urllib.request.urlopen(request, timeout=60) as source:
                    while block := source.read(1024 * 1024):
                        digest.update(block)
                        size += len(block)
                        target.write(block)
                if size != part["bytes"] or digest.hexdigest() != part["sha256"]:
                    raise RuntimeError(f"Data part checksum mismatch: {index + 1}")
        temporary.replace(path)
    if path.stat().st_size != metadata["bytes"] or checksum(path) != metadata["sha256"]:
        raise RuntimeError(f"Data archive checksum mismatch: {path}")
    prefix = metadata["archive_prefix"].rstrip("/")
    count = 0
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            name = PurePosixPath(member.name)
            parts = name.parts
            if name.is_absolute() or ".." in parts or not parts or parts[0] != prefix:
                raise RuntimeError(f"Unexpected archive path: {member.name}")
            if member.isdir():
                continue
            if not member.isfile():
                raise RuntimeError(f"Unexpected non-file archive member: {member.name}")
            relative = PurePosixPath(*parts[1:])
            if not (relative.parts and relative.parts[0] == "data") and str(relative) != "reference/inference_manifest.json":
                raise RuntimeError(f"Archive cannot replace source files: {relative}")
            target = ROOT.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
            count += 1
    print(f"Verified SHA256 and unpacked {count} files into {ROOT}", flush=True)
    print("Next: python reproduce.py all --output outputs" if args.kind == "replay" else "Next: python infer.py verify-data", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
