#!/usr/bin/env python3
"""Package original VitaminC and one prepared split directory, with SHA256 verification."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def sha256_file(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else hashlib.sha256(handle.read()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('data'))
    parser.add_argument('--splits', default='vitaminc_splits_v2')
    parser.add_argument('--archive', default='vitaminc_v2.zip')
    args = parser.parse_args()
    if Path(args.splits).name != args.splits or Path(args.archive).name != args.archive:
        parser.error('--splits and --archive must be names within --data-root')
    root = args.data_root
    files = [p for directory in (root / 'vitaminc', root / args.splits)
             for p in sorted(directory.iterdir()) if p.is_file() and not p.name.startswith('.')
             and (p.suffix in ('.jsonl', '.json', '.txt') or p.name == 'LICENSE')]
    expected = {f'vitaminc/{n}' for n in ('train.jsonl', 'dev.jsonl', 'test.jsonl', 'LICENSE', 'README.txt')}
    expected |= {f'{args.splits}/{n}' for n in ('prompt_dev.jsonl', 'calibration.jsonl', 'threshold_validation.jsonl', 'test.jsonl', 'split_report.json')}
    if {p.relative_to(root).as_posix() for p in files} != expected:
        raise ValueError('Missing or unexpected archive source files')
    entries = [dict(path=p.relative_to(root).as_posix(), bytes=p.stat().st_size, sha256=sha256_file(p)) for p in files]
    archive = root / args.archive
    temp = archive.with_suffix(archive.suffix + '.tmp')
    with zipfile.ZipFile(temp, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in files:
            z.write(p, p.relative_to(root).as_posix())
    with zipfile.ZipFile(temp) as z:
        if z.testzip() is not None:
            raise ValueError('ZIP CRC check failed')
        for entry in entries:
            with z.open(entry['path']) as handle:
                h = hashlib.sha256()
                for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                    h.update(chunk)
            if h.hexdigest() != entry['sha256']:
                raise ValueError('Archived bytes differ from source')
    temp.replace(archive)
    manifest = dict(archive=archive.as_posix(), archive_bytes=archive.stat().st_size,
                    archive_sha256=sha256_file(archive), files=entries,
                    extract_from_repo_root=f'unzip {archive.as_posix()} -d {root.as_posix()}')
    path = archive.with_name(archive.stem + '_archive_manifest.json')
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in manifest.items() if k != 'files'}, indent=2))


if __name__ == '__main__':
    main()
