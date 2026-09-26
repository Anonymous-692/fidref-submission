"""Restore bundled results and verify every record using the release manifest."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parent


def main():
    files = json.loads((ROOT / 'RELEASE_MANIFEST.json').read_text())['files']
    expected = {p: v for p, v in files.items() if p.startswith('results/')}
    seen = set()
    with tarfile.open(ROOT / 'results.tar.xz', mode='r|xz') as archive:
        for member in archive:
            name = member.name
            relative = PurePosixPath(name)
            if (name not in expected or name in seen or relative.is_absolute()
                    or '..' in relative.parts or not member.isfile()):
                raise ValueError('Unexpected archive entry')
            entry = expected[name]
            if member.size != entry['bytes']:
                raise ValueError('Archive size mismatch: ' + name)
            path = ROOT / name
            for parent in (path, *path.parents):
                if parent == ROOT:
                    break
                if parent.is_symlink():
                    raise ValueError('Symlink in extraction path')
            path.parent.mkdir(parents=True, exist_ok=True)
            content = archive.extractfile(member).read()
            if hashlib.sha256(content).hexdigest() != entry['sha256']:
                raise ValueError('Archive checksum mismatch: ' + name)
            if path.exists():
                if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
                    raise ValueError('Existing file differs; preserve it before restoring: ' + name)
            else:
                temporary = None
                try:
                    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
                        temporary = Path(stream.name)
                        stream.write(content)
                    os.replace(temporary, path)
                finally:
                    if temporary and temporary.exists():
                        temporary.unlink()
            seen.add(name)
    if seen != set(expected):
        raise ValueError('Missing archive records')
    print(f'Restored and verified {len(seen)} result files.')


if __name__ == '__main__':
    main()
