"""Download Spider: question/SQL pairs AND the SQLite databases.

Two separate sources, because no single public artifact carries both:

1. `xlangai/spider` -- parquet with `db_id` / `question` / `query`. 7,000 train, 1,034
   validation. About 1 MB. This is the canonical Hub distribution and it contains no
   databases at all.
2. `HAL-9001/spider-databases` -- `spider_data.zip` (~206 MB), the databases plus
   `tables.json`. Needed because execution accuracy means actually running the SQL.

Both are public and ungated; no Hugging Face token is required.

    python scripts/download_data.py
    python scripts/download_data.py --force     # re-download even if present
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from text2sql_lora.config import Paths

PAIRS_REPO = "xlangai/spider"
PAIRS_FILES = {
    "train": "spider/train-00000-of-00001.parquet",
    "validation": "spider/validation-00000-of-00001.parquet",
}
DATABASES_REPO = "HAL-9001/spider-databases"
DATABASES_FILE = "spider_data.zip"


def download_pairs(paths: Paths, force: bool) -> None:
    target_dir = paths.data_dir / "spider_pairs"
    target_dir.mkdir(parents=True, exist_ok=True)

    for split, remote_name in PAIRS_FILES.items():
        target = paths.parquet(split)
        if target.exists() and not force:
            print(f"[pairs] {split}: already present ({target.stat().st_size / 1e3:.0f} kB)")
            continue
        print(f"[pairs] {split}: downloading {remote_name} from {PAIRS_REPO}")
        cached = hf_hub_download(
            repo_id=PAIRS_REPO, filename=remote_name, repo_type="dataset"
        )
        shutil.copyfile(cached, target)
        print(f"[pairs] {split}: wrote {target}")


def download_databases(paths: Paths, force: bool) -> None:
    spider_dir = paths.spider_dir()
    if paths.tables_json().exists() and paths.database_dir().is_dir() and not force:
        n_dbs = sum(1 for _ in paths.database_dir().iterdir() if _.is_dir())
        print(f"[databases] already present ({n_dbs} databases)")
        return

    print(f"[databases] downloading {DATABASES_FILE} (~206 MB) from {DATABASES_REPO}")
    archive = hf_hub_download(
        repo_id=DATABASES_REPO, filename=DATABASES_FILE, repo_type="dataset"
    )

    print(f"[databases] extracting into {paths.data_dir}")
    paths.data_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(paths.data_dir)

    # The archive may or may not nest everything under a `spider_data/` directory
    # depending on how it was packed. Normalise so `Paths` finds it either way.
    if not paths.tables_json().exists():
        found = next(paths.data_dir.rglob("tables.json"), None)
        if found is None:
            raise SystemExit(
                f"extracted {DATABASES_FILE} but found no tables.json under {paths.data_dir}"
            )
        if found.parent != spider_dir:
            print(f"[databases] normalising layout: {found.parent} -> {spider_dir}")
            if spider_dir.exists():
                shutil.rmtree(spider_dir)
            shutil.move(str(found.parent), str(spider_dir))


def verify(paths: Paths) -> int:
    """Check that the two downloads agree with each other. Returns an exit code."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from text2sql_lora.data.spider import load_schemas, load_split, missing_databases

    schemas = load_schemas(paths)
    print(f"\n[verify] tables.json: {len(schemas)} databases")

    problems = 0
    for split in ("train", "validation"):
        data = load_split(split, paths=paths, schemas=schemas)
        n_dbs = len(set(data.db_ids()))
        missing = missing_databases(data, paths=paths)
        print(f"[verify] {split}: {len(data)} examples over {n_dbs} databases")
        if missing:
            problems += 1
            print(f"[verify] {split}: MISSING {len(missing)} .sqlite files: {missing[:5]}")

    train = load_split("train", paths=paths, schemas=schemas)
    dev = load_split("validation", paths=paths, schemas=schemas)
    overlap = set(train.db_ids()) & set(dev.db_ids())
    if overlap:
        problems += 1
        print(f"[verify] WARNING: {len(overlap)} databases appear in BOTH train and dev: {overlap}")
    else:
        print("[verify] train and dev databases are disjoint (as Spider intends)")

    return 1 if problems else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()

    paths = Paths()
    paths.data_dir.mkdir(parents=True, exist_ok=True)

    download_pairs(paths, args.force)
    download_databases(paths, args.force)

    if args.skip_verify:
        return 0
    return verify(paths)


if __name__ == "__main__":
    raise SystemExit(main())
