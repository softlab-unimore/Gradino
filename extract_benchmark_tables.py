#!/usr/bin/env python3

import csv
import itertools
import os
import shutil
import sqlite3
import subprocess
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import duckdb
import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path("benchmark_extraction")
DOWNLOAD_DIR = BASE_DIR / "downloads"
UNPACK_DIR = BASE_DIR / "unpacked"
DB_DIR = BASE_DIR / "materialized_dbs"
EXPORT_DIR = BASE_DIR / "exports"
TABLE_CSV_DIR = EXPORT_DIR / "tables_csv"

DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
UNPACK_DIR.mkdir(parents=True, exist_ok=True)
DB_DIR.mkdir(parents=True, exist_ok=True)
EXPORT_DIR.mkdir(parents=True, exist_ok=True)
TABLE_CSV_DIR.mkdir(parents=True, exist_ok=True)

DATASET_URLS = {
    "spider": "https://github.com/lucacontalbo/dbs/releases/download/v1/spider_data.zip",
    "beaver": "https://github.com/lucacontalbo/dbs/releases/download/v1/dw_csvs.zip",
    "bird": "https://github.com/lucacontalbo/dbs/releases/download/v1/dev.zip",
}

MIN_ROWS = 3
MIN_COLS = 3
MIN_DENSITY = 0.95

MAX_ROW_ATTRS = 4
MAX_COL_ATTRS = 4
MAX_UNIQUE_PER_AXIS_ATTR = 60
MAX_ESTIMATED_CARTESIAN_CELLS = 20000

# If None, keep all valid pivots found for a relation.
MAX_RESULTS_PER_RELATION = None

# BEAVER CSV materialization: limit how many wide CSV joins we attempt.
MAX_BEAVER_ROWS_PER_TABLE = None  # e.g. 200000 if needed


# ============================================================
# DOWNLOAD / UNZIP
# ============================================================

def download_file(url: str, dst: Path) -> None:
    if dst.exists():
        print(f"Already downloaded: {dst}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {dst}")
    urllib.request.urlretrieve(url, dst)


def unzip_file(zip_path: Path, extract_dir: Path) -> None:
    marker = extract_dir / ".unzipped_ok"
    if marker.exists():
        print(f"Already unzipped: {extract_dir}")
        return
    extract_dir.mkdir(parents=True, exist_ok=True)
    print(f"Unzipping {zip_path} -> {extract_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    marker.write_text("ok", encoding="utf-8")


# ============================================================
# RELATION DISCOVERY
# ============================================================

@dataclass
class RelationTarget:
    dataset: str
    db_name: str
    relation_name: str
    relation_type: str  # "table" or "view"
    db_kind: str        # "sqlite" or "duckdb"
    db_path: Path


def find_files(root: Path, suffixes: Iterable[str]) -> List[Path]:
    suffixes = tuple(s.lower() for s in suffixes)
    out = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in suffixes:
            out.append(p)
    return sorted(out)


def sqlite_relation_targets(db_path: Path, dataset: str) -> List[RelationTarget]:
    targets: List[RelationTarget] = []
    db_name = db_path.stem
    con = sqlite3.connect(db_path)
    try:
        q = """
            SELECT name, type
            FROM sqlite_master
            WHERE type IN ('table', 'view')
              AND name NOT LIKE 'sqlite_%'
            ORDER BY type, name
        """
        rows = pd.read_sql_query(q, con)
        for _, row in rows.iterrows():
            targets.append(
                RelationTarget(
                    dataset=dataset,
                    db_name=db_name,
                    relation_name=row["name"],
                    relation_type=row["type"],
                    db_kind="sqlite",
                    db_path=db_path,
                )
            )
    finally:
        con.close()
    return targets


def duckdb_relation_targets(db_path: Path, dataset: str) -> List[RelationTarget]:
    targets: List[RelationTarget] = []
    db_name = db_path.stem
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        rows = con.execute("""
            SELECT table_schema, table_name
            FROM information_schema.tables
            WHERE table_schema NOT IN ('information_schema', 'pg_catalog')
            ORDER BY table_schema, table_name
        """).fetchdf()
        for _, row in rows.iterrows():
            schema = row["table_schema"]
            table = row["table_name"]
            relation_name = f"{schema}.{table}"
            targets.append(
                RelationTarget(
                    dataset=dataset,
                    db_name=db_name,
                    relation_name=relation_name,
                    relation_type="table",
                    db_kind="duckdb",
                    db_path=db_path,
                )
            )
    finally:
        con.close()
    return targets


def read_relation(target: RelationTarget) -> pd.DataFrame:
    if target.db_kind == "sqlite":
        con = sqlite3.connect(target.db_path)
        try:
            return pd.read_sql_query(f'SELECT * FROM "{target.relation_name}"', con)
        finally:
            con.close()

    if target.db_kind == "duckdb":
        con = duckdb.connect(str(target.db_path), read_only=True)
        try:
            return con.execute(f'SELECT * FROM {target.relation_name}').fetchdf()
        finally:
            con.close()

    raise ValueError(f"Unsupported db_kind: {target.db_kind}")


# ============================================================
# BEAVER CSV -> SQLITE MATERIALIZATION
# ============================================================

def safe_sqlite_ident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def sanitize_name(name: str) -> str:
    out = []
    for ch in str(name):
        if ch.isalnum() or ch == "_":
            out.append(ch)
        else:
            out.append("_")
    name2 = "".join(out).strip("_")
    if not name2:
        name2 = "unnamed"
    if name2[0].isdigit():
        name2 = f"t_{name2}"
    return name2


def materialize_csv_group_to_sqlite(csv_files: List[Path], out_db: Path) -> None:
    if out_db.exists():
        print(f"Already materialized: {out_db}")
        return

    print(f"Materializing CSV group -> {out_db}")
    con = sqlite3.connect(out_db)
    try:
        for csv_path in csv_files:
            table_name = sanitize_name(csv_path.stem)
            print(f"  Importing CSV as table {table_name}: {csv_path.name}")
            try:
                df = pd.read_csv(csv_path, low_memory=False)
            except UnicodeDecodeError:
                df = pd.read_csv(csv_path, encoding="latin1", low_memory=False)

            if MAX_BEAVER_ROWS_PER_TABLE is not None and len(df) > MAX_BEAVER_ROWS_PER_TABLE:
                df = df.iloc[:MAX_BEAVER_ROWS_PER_TABLE].copy()

            df.columns = [sanitize_name(c) for c in df.columns]
            df.to_sql(table_name, con, index=False, if_exists="replace")
    finally:
        con.close()


def group_csvs_for_beaver(root: Path) -> Dict[str, List[Path]]:
    csvs = find_files(root, [".csv"])
    groups: Dict[str, List[Path]] = {}
    for p in csvs:
        # group by immediate parent dir if possible; otherwise by stem prefix
        key = p.parent.name if p.parent != root else "root_csvs"
        groups.setdefault(key, []).append(p)
    return groups


# ============================================================
# PIVOT SEARCH
# ============================================================

def numeric_conversion(series: pd.Series) -> Tuple[pd.Series, float]:
    numeric = pd.to_numeric(series, errors="coerce")
    coverage = float(numeric.notna().mean())
    return numeric, coverage


def is_float_attribute(series: pd.Series) -> bool:
    numeric, coverage = numeric_conversion(series)
    numeric = numeric.dropna()
    if coverage < 0.95 or numeric.empty:
        return False
    return bool(((numeric % 1) != 0).any())


def is_integer_attribute(series: pd.Series) -> bool:
    numeric, coverage = numeric_conversion(series)
    numeric = numeric.dropna()
    if coverage < 0.95 or numeric.empty:
        return False
    return bool(np.allclose(numeric, np.round(numeric)))


def is_axis_attribute(series: pd.Series) -> bool:
    non_null = series.dropna()
    nunique = non_null.nunique()

    if nunique < 3:
        return False
    if nunique > MAX_UNIQUE_PER_AXIS_ATTR:
        return False

    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        return True
    if is_integer_attribute(series):
        return True
    if pd.api.types.is_bool_dtype(series):
        return True
    return False


def normalize_axis_value(v: Any) -> Any:
    if pd.isna(v):
        return None
    return v


def candidate_axis_combinations(cols: List[str], max_size: int) -> List[Tuple[str, ...]]:
    combos: List[Tuple[str, ...]] = []
    for size in range(1, min(max_size, len(cols)) + 1):
        combos.extend(itertools.combinations(cols, size))
    return combos


def disjoint(a: Sequence[str], b: Sequence[str]) -> bool:
    return set(a).isdisjoint(set(b))


def pivot_density(pivot: pd.DataFrame) -> float:
    return float(pivot.notna().to_numpy().mean())


def estimated_cartesian_cells(df: pd.DataFrame, row_attrs: Sequence[str], col_attrs: Sequence[str]) -> int:
    row_prod = 1
    for c in row_attrs:
        row_prod *= max(1, int(df[c].dropna().nunique()))
    col_prod = 1
    for c in col_attrs:
        col_prod *= max(1, int(df[c].dropna().nunique()))
    return row_prod * col_prod


def score_result(result: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        result["non_nan_density"],
        result["num_row_attributes"] + result["num_column_attributes"],
        min(result["num_row_attributes"], result["num_column_attributes"]),
        result["pivot_cells"],
        result["pivot_rows"],
        result["pivot_columns"],
    )


def find_all_valid_pivots(df: pd.DataFrame, target: RelationTarget) -> List[Dict[str, Any]]:
    if df.shape[0] < 9 or df.shape[1] < 3:
        return []

    work_df = df.copy()

    value_candidates: List[str] = []
    axis_candidates: List[str] = []

    for col in work_df.columns:
        s = work_df[col]
        non_null = s.dropna()
        if non_null.empty or non_null.nunique() < 3:
            continue

        if is_float_attribute(s):
            value_candidates.append(col)
        elif is_axis_attribute(s):
            axis_candidates.append(col)

    # Need at least 3 axis attrs because we only allow 2x1 / 1x2 / 2x2
    if not value_candidates or len(axis_candidates) < 3:
        return []

    # Low-cardinality attrs first: more likely to form dense pivots
    axis_candidates = sorted(
        axis_candidates,
        key=lambda c: (work_df[c].dropna().nunique(), c)
    )

    # Hard cap for speed. Increase a bit if you want more recall.
    MAX_AXIS_CANDIDATES_TO_USE = 10
    axis_candidates = axis_candidates[:MAX_AXIS_CANDIDATES_TO_USE]

    # Only search these shapes; each satisfies "at least one side has depth >= 2"
    shape_plan = [
        (2, 1),
        (1, 2),
        (2, 2),
    ]

    results: List[Dict[str, Any]] = []
    seen_keys = set()

    for value_col in value_candidates:
        base_cols = axis_candidates + [value_col]
        base_df = work_df[base_cols].dropna(subset=[value_col]).copy()
        if base_df.empty:
            continue

        full_distinct_cache: Dict[Tuple[str, ...], int] = {}

        def distinct_count_full(cols: Sequence[str]) -> int:
            key = tuple(cols)
            if key not in full_distinct_cache:
                full_distinct_cache[key] = len(base_df[list(cols)].drop_duplicates())
            return full_distinct_cache[key]

        for row_size, col_size in shape_plan:
            if len(axis_candidates) < row_size + col_size:
                continue

            row_combos = list(itertools.combinations(axis_candidates, row_size))
            row_combos.sort(
                key=lambda combo: (
                    np.prod([max(1, base_df[c].dropna().nunique()) for c in combo]),
                    combo,
                )
            )

            for row_attrs in row_combos:
                row_key_count = distinct_count_full(row_attrs)
                if row_key_count < MIN_ROWS:
                    continue

                remaining = [c for c in axis_candidates if c not in row_attrs]
                if len(remaining) < col_size:
                    continue

                col_combos = list(itertools.combinations(remaining, col_size))
                col_combos.sort(
                    key=lambda combo: (
                        np.prod([max(1, base_df[c].dropna().nunique()) for c in combo]),
                        combo,
                    )
                )

                for col_attrs in col_combos:
                    col_key_count = distinct_count_full(col_attrs)
                    if col_key_count < MIN_COLS:
                        continue

                    total_cells = row_key_count * col_key_count
                    if total_cells > MAX_ESTIMATED_CARTESIAN_CELLS:
                        continue

                    joint_attrs = list(row_attrs) + list(col_attrs)
                    filled_cells_upper = distinct_count_full(joint_attrs)
                    if filled_cells_upper / total_cells < MIN_DENSITY:
                        continue

                    needed_cols = joint_attrs + [value_col]
                    sub = base_df[needed_cols].dropna().copy()
                    if sub.empty:
                        continue

                    # Local cache on the filtered subset
                    sub_distinct_cache: Dict[Tuple[str, ...], int] = {}

                    def distinct_count_sub(cols: Sequence[str]) -> int:
                        key = tuple(cols)
                        if key not in sub_distinct_cache:
                            sub_distinct_cache[key] = len(sub[list(cols)].drop_duplicates())
                        return sub_distinct_cache[key]

                    row_keys_sub = distinct_count_sub(row_attrs)
                    col_keys_sub = distinct_count_sub(col_attrs)

                    if row_keys_sub < MIN_ROWS or col_keys_sub < MIN_COLS:
                        continue

                    total_cells_sub = row_keys_sub * col_keys_sub
                    if total_cells_sub > MAX_ESTIMATED_CARTESIAN_CELLS:
                        continue

                    filled_cells_sub = distinct_count_sub(joint_attrs)
                    density_sub = filled_cells_sub / total_cells_sub
                    if density_sub < MIN_DENSITY:
                        continue

                    try:
                        pivot = sub.pivot_table(
                            index=list(row_attrs),
                            columns=list(col_attrs),
                            values=value_col,
                            aggfunc="mean",
                            observed=False,
                        )
                    except Exception:
                        continue

                    if pivot.shape[0] < MIN_ROWS or pivot.shape[1] < MIN_COLS:
                        continue

                    actual_density = pivot_density(pivot)
                    if actual_density < MIN_DENSITY:
                        continue

                    result = {
                        "dataset": target.dataset,
                        "db_name": target.db_name,
                        "relation": target.relation_name,
                        "relation_type": target.relation_type,
                        "db_kind": target.db_kind,
                        "row_attributes": list(row_attrs),
                        "column_attributes": list(col_attrs),
                        "value_attribute": value_col,
                        "num_row_attributes": len(row_attrs),
                        "num_column_attributes": len(col_attrs),
                        "pivot_rows": int(pivot.shape[0]),
                        "pivot_columns": int(pivot.shape[1]),
                        "pivot_cells": int(pivot.shape[0] * pivot.shape[1]),
                        "non_nan_density": actual_density,
                        "source_rows": int(df.shape[0]),
                        "source_columns": int(df.shape[1]),
                    }

                    key = (
                        result["dataset"],
                        result["db_name"],
                        result["relation"],
                        tuple(result["row_attributes"]),
                        tuple(result["column_attributes"]),
                        result["value_attribute"],
                    )
                    if key in seen_keys:
                        continue

                    seen_keys.add(key)
                    results.append(result)

                    if len(results) >= 2:
                        results.sort(key=score_result, reverse=True)
                        return results

    results.sort(key=score_result, reverse=True)
    return results


def export_relation_csv(df: pd.DataFrame, target: RelationTarget) -> str:
    safe_name = f"{target.dataset}__{sanitize_name(target.db_name)}__{sanitize_name(target.relation_name)}.csv"
    out_path = TABLE_CSV_DIR / safe_name
    if not out_path.exists():
        df.to_csv(out_path, index=False)
    return str(out_path)


# ============================================================
# DATASET PREP
# ============================================================

def prepare_spider(spider_root: Path) -> List[RelationTarget]:
    db_files = find_files(spider_root, [".sqlite", ".db"])
    targets: List[RelationTarget] = []
    for db_path in db_files:
        try:
            targets.extend(sqlite_relation_targets(db_path, dataset="spider"))
        except Exception as e:
            print(f"SKIP spider db {db_path}: {e}")
    return targets


def prepare_bird(bird_root: Path) -> List[RelationTarget]:
    sqlite_files = find_files(bird_root, [".sqlite", ".db"])
    duckdb_files = find_files(bird_root, [".duckdb"])

    targets: List[RelationTarget] = []
    for db_path in sqlite_files:
        try:
            targets.extend(sqlite_relation_targets(db_path, dataset="bird"))
        except Exception as e:
            print(f"SKIP bird sqlite db {db_path}: {e}")

    for db_path in duckdb_files:
        try:
            targets.extend(duckdb_relation_targets(db_path, dataset="bird"))
        except Exception as e:
            print(f"SKIP bird duckdb db {db_path}: {e}")

    return targets


def prepare_beaver(beaver_root: Path) -> List[RelationTarget]:
    # If there are ready DBs, use them directly.
    sqlite_files = find_files(beaver_root, [".sqlite", ".db"])
    duckdb_files = find_files(beaver_root, [".duckdb"])

    targets: List[RelationTarget] = []

    for db_path in sqlite_files:
        try:
            targets.extend(sqlite_relation_targets(db_path, dataset="beaver"))
        except Exception as e:
            print(f"SKIP beaver sqlite db {db_path}: {e}")

    for db_path in duckdb_files:
        try:
            targets.extend(duckdb_relation_targets(db_path, dataset="beaver"))
        except Exception as e:
            print(f"SKIP beaver duckdb db {db_path}: {e}")

    if targets:
        return targets

    # Otherwise materialize CSV groups into SQLite DBs.
    groups = group_csvs_for_beaver(beaver_root)
    for group_name, csv_files in groups.items():
        if not csv_files:
            continue
        out_db = DB_DIR / f"beaver_{sanitize_name(group_name)}.sqlite"
        try:
            materialize_csv_group_to_sqlite(csv_files, out_db)
            targets.extend(sqlite_relation_targets(out_db, dataset="beaver"))
        except Exception as e:
            print(f"SKIP beaver group {group_name}: {e}")

    return targets


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    downloaded: Dict[str, Path] = {}
    unpacked: Dict[str, Path] = {}

    for name, url in DATASET_URLS.items():
        zip_path = DOWNLOAD_DIR / f"{name}.zip"
        download_file(url, zip_path)
        downloaded[name] = zip_path

        extract_dir = UNPACK_DIR / name
        unzip_file(zip_path, extract_dir)
        unpacked[name] = extract_dir

    all_targets: List[RelationTarget] = []
    all_targets.extend(prepare_spider(unpacked["spider"]))
    all_targets.extend(prepare_bird(unpacked["bird"]))
    all_targets.extend(prepare_beaver(unpacked["beaver"]))

    print(f"\nTotal relations discovered: {len(all_targets)}")

    summary_rows: List[Dict[str, Any]] = []
    tried_relations = 0
    good_relations = 0

    for target in all_targets:
        print(f"Scanning {target.dataset} | {target.db_name} | {target.relation_name}")
        try:
            df = read_relation(target)
        except Exception as e:
            print(f"  SKIP read failure: {e}")
            continue

        tried_relations += 1

        try:
            results = find_all_valid_pivots(df, target)
        except Exception as e:
            print(f"  SKIP extraction failure: {e}")
            continue

        if results:
            good_relations += 1
            csv_path = export_relation_csv(df, target)
            for result in results:
                result["table_csv_path"] = csv_path
                summary_rows.append(result)
                print(
                    f"  PASS: value={result['value_attribute']} "
                    f"rows={result['row_attributes']} "
                    f"cols={result['column_attributes']} "
                    f"shape={result['pivot_rows']}x{result['pivot_columns']} "
                    f"levels=({result['num_row_attributes']},{result['num_column_attributes']}) "
                    f"density={result['non_nan_density']:.3f}"
                )

    if not summary_rows:
        print("\nNo qualifying relations found.")
        print(f"Total relations tried: {tried_relations}")
        print(f"Total good relations: {good_relations}")
        return

    summary_df = pd.DataFrame(summary_rows)
    summary_df["row_attributes"] = summary_df["row_attributes"].apply(lambda x: "; ".join(x))
    summary_df["column_attributes"] = summary_df["column_attributes"].apply(lambda x: "; ".join(x))
    summary_df["total_header_levels"] = (
        summary_df["num_row_attributes"] + summary_df["num_column_attributes"]
    )
    summary_df["min_header_side_levels"] = summary_df[
        ["num_row_attributes", "num_column_attributes"]
    ].min(axis=1)

    summary_df = summary_df.sort_values(
        by=[
            "non_nan_density",
            "total_header_levels",
            "min_header_side_levels",
            "pivot_cells",
            "pivot_rows",
            "pivot_columns",
            "dataset",
            "db_name",
            "relation",
            "value_attribute",
            "row_attributes",
            "column_attributes",
        ],
        ascending=[False, False, False, False, False, False, True, True, True, True, True, True],
    ).reset_index(drop=True)

    summary_csv = EXPORT_DIR / "qualifying_dense_pivots_benchmarks.csv"
    summary_df.to_csv(summary_csv, index=False)

    best_per_relation = (
        summary_df.sort_values(
            by=[
                "non_nan_density",
                "total_header_levels",
                "min_header_side_levels",
                "pivot_cells",
                "pivot_rows",
                "pivot_columns",
            ],
            ascending=[False, False, False, False, False, False],
        )
        .groupby(["dataset", "db_name", "relation"], as_index=False)
        .head(1)
        .reset_index(drop=True)
    )

    best_per_relation_csv = EXPORT_DIR / "best_qualifying_dense_pivots_per_relation_benchmarks.csv"
    best_per_relation.to_csv(best_per_relation_csv, index=False)

    stats = pd.DataFrame(
        [
            {"metric": "total_relations_tried", "value": tried_relations},
            {"metric": "total_good_relations", "value": good_relations},
            {"metric": "total_valid_pivots", "value": len(summary_df)},
            {
                "metric": "total_datasets_with_good_relations",
                "value": summary_df[["dataset"]].drop_duplicates().shape[0],
            },
            {
                "metric": "total_databases_with_good_relations",
                "value": summary_df[["dataset", "db_name"]].drop_duplicates().shape[0],
            },
        ]
    )
    stats_csv = EXPORT_DIR / "stats.csv"
    stats.to_csv(stats_csv, index=False)

    print("\nDone.")
    print(f"Saved full summary: {summary_csv}")
    print(f"Saved best-per-relation summary: {best_per_relation_csv}")
    print(f"Saved stats: {stats_csv}")
    print(f"Total relations tried: {tried_relations}")
    print(f"Total good relations: {good_relations}")
    print(f"Total valid pivots: {len(summary_df)}")


if __name__ == "__main__":
    main()
