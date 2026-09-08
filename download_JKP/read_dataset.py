"""Cleaning and loading utilities for the locally downloaded JKP USA data.

The raw data are stored one year per parquet file in ``data/JKP_USA``. On
the first call, :func:`load_jkp_value` builds a cleaned cache next to that
directory. Later calls read the cache directly and load only requested columns.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from pandas.api.types import is_numeric_dtype


META_COLUMNS = [
    "id",
    "eom",
    "excntry",
    "gvkey",
    "permno",
    "size_grp",
    "me",
    "ret_exc_lead1m",
]

_REQUIRED_COLUMNS = ["id", "eom", "permno", "me", "ret_exc_lead1m"]
_START_DATE = pd.Timestamp("1963-01-01")
_END_DATE = pd.Timestamp("2024-12-31")
_MAX_CHARACTERISTIC_MISSING_SHARE = 1.0 / 3.0
_MAX_ROW_MISSING_SHARE = 0.3
_CLEANING_VERSION = 2
_MANIFEST_NAME = "cleaning_manifest.json"


def _as_characteristic_list(characteristics):
    if characteristics is None:
        return []
    if isinstance(characteristics, str):
        return [characteristics]
    return list(characteristics)


def _raw_files(data_dir: Path) -> list[Path]:
    files = sorted(data_dir.glob("JKP_USA_*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No JKP parquet files found in: {data_dir.resolve()}"
        )
    return files


def _source_signature(files: list[Path]) -> list[dict[str, int | str]]:
    return [
        {
            "name": file.name,
            "size": file.stat().st_size,
            "mtime_ns": file.stat().st_mtime_ns,
        }
        for file in files
    ]


def _read_manifest(clean_dir: Path) -> dict | None:
    manifest_path = clean_dir / _MANIFEST_NAME
    if not manifest_path.exists():
        return None
    try:
        with manifest_path.open(encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, json.JSONDecodeError):
        return None


def _cache_is_current(
    manifest: dict | None,
    source_signature: list[dict[str, int | str]],
    clean_dir: Path,
) -> bool:
    if manifest is None:
        return False
    if manifest.get("cleaning_version") != _CLEANING_VERSION:
        return False
    if manifest.get("source_files") != source_signature:
        return False

    output_files = manifest.get("output_files", [])
    return bool(output_files) and all((clean_dir / name).exists() for name in output_files)


def _coerce_characteristics(frame: pd.DataFrame, characteristics: list[str]) -> None:
    """Normalize rare all-missing/string parquet columns to numeric in place."""
    for characteristic in characteristics:
        if not is_numeric_dtype(frame[characteristic]):
            frame[characteristic] = pd.to_numeric(
                frame[characteristic], errors="coerce"
            )


def _sample_mask(frame: pd.DataFrame) -> pd.Series:
    dates = pd.to_datetime(frame["eom"])
    return (
        dates.between(_START_DATE, _END_DATE)
        & frame["size_grp"].notna()
        & frame["size_grp"].ne("nano")
    )


def _rank_standardize_by_month(
    frame: pd.DataFrame, characteristics: list[str]
) -> None:
    """Rank each monthly cross-section to [-0.5, 0.5] and impute at zero."""
    grouped = frame.groupby("eom", sort=False)[characteristics]
    ranks = grouped.rank(method="average", na_option="keep")
    counts = grouped.transform("count")
    standardized = (ranks - 1.0).div(counts - 1.0) - 0.5
    standardized = standardized.mask(counts <= 1).fillna(0.0)
    frame.loc[:, characteristics] = standardized


def build_clean_jkp_cache(
    data_dir,
    clean_dir=None,
    *,
    force: bool = False,
) -> tuple[Path, dict]:
    """Build or reuse the cleaned, annual JKP parquet cache."""
    data_dir = Path(data_dir)
    clean_dir = (
        Path(clean_dir)
        if clean_dir is not None
        else data_dir.with_name(f"{data_dir.name}_clean")
    )
    files = _raw_files(data_dir)
    source_signature = _source_signature(files)
    manifest = _read_manifest(clean_dir)

    if not force and _cache_is_current(manifest, source_signature, clean_dir):
        return clean_dir, manifest

    first_columns = pd.read_parquet(files[0]).columns.tolist()
    missing_meta = [column for column in META_COLUMNS if column not in first_columns]
    if missing_meta:
        raise ValueError(f"Missing required JKP metadata columns: {missing_meta}")

    characteristics = [column for column in first_columns if column not in META_COLUMNS]
    non_missing = pd.Series(0, index=characteristics, dtype="int64")
    observation_count = 0

    print("Building cleaned JKP cache (coverage pass)...")
    coverage_columns = ["eom", "size_grp", *characteristics]
    for file in files:
        frame = pd.read_parquet(file, columns=coverage_columns)
        _coerce_characteristics(frame, characteristics)
        frame = frame.loc[_sample_mask(frame), characteristics]
        if frame.empty:
            continue
        observation_count += len(frame)
        non_missing = non_missing.add(frame.notna().sum(), fill_value=0)

    if observation_count == 0:
        raise ValueError("No observations remain after the date and size filters.")

    missing_rate = 1.0 - non_missing / observation_count
    kept_characteristics = missing_rate[
        missing_rate <= _MAX_CHARACTERISTIC_MISSING_SHARE
    ].index.tolist()
    if not kept_characteristics:
        raise ValueError("No characteristics pass the global missing-value screen.")

    print(
        "Characteristics after <=1/3 missing filter: "
        f"{len(kept_characteristics)} (target: 132)"
    )
    print("Cleaning, rank-standardizing, and writing annual cache files...")
    clean_dir.mkdir(parents=True, exist_ok=True)
    output_files: list[str] = []
    total_rows = 0
    load_columns = [*META_COLUMNS, *kept_characteristics]
    max_missing_count = len(kept_characteristics) * _MAX_ROW_MISSING_SHARE

    for file in files:
        frame = pd.read_parquet(file, columns=load_columns)
        _coerce_characteristics(frame, kept_characteristics)
        frame["eom"] = pd.to_datetime(frame["eom"])
        frame = frame.loc[_sample_mask(frame)].copy()
        if frame.empty:
            continue

        row_missing = frame[kept_characteristics].isna().sum(axis=1)
        frame = frame.loc[row_missing <= max_missing_count].copy()
        if frame.empty:
            continue

        _rank_standardize_by_month(frame, kept_characteristics)
        frame = frame.sort_values(["id", "eom"], kind="stable").reset_index(drop=True)

        if frame[kept_characteristics].isna().any().any():
            raise AssertionError(f"NaNs remain in characteristics for {file.name}")

        output_name = file.name.replace("JKP_USA_", "JKP_USA_clean_", 1)
        output_path = clean_dir / output_name
        temporary_path = output_path.with_suffix(".parquet.tmp")
        frame.to_parquet(temporary_path, index=False, compression="zstd")
        temporary_path.replace(output_path)
        output_files.append(output_name)
        total_rows += len(frame)
        print(f"  {file.stem}: {len(frame):,} cleaned stock-months")

    manifest = {
        "cleaning_version": _CLEANING_VERSION,
        "source_files": source_signature,
        "output_files": output_files,
        "date_start": _START_DATE.strftime("%Y-%m-%d"),
        "date_end": _END_DATE.strftime("%Y-%m-%d"),
        "max_characteristic_missing_share": _MAX_CHARACTERISTIC_MISSING_SHARE,
        "max_row_missing_share": _MAX_ROW_MISSING_SHARE,
        "rows_after_cleaning": total_rows,
        "kept_characteristics": kept_characteristics,
        "missing_rates": {key: float(value) for key, value in missing_rate.items()},
    }
    manifest_path = clean_dir / _MANIFEST_NAME
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    with temporary_manifest.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
    temporary_manifest.replace(manifest_path)

    print(f"Cleaned cache saved in: {clean_dir.resolve()}")
    return clean_dir, manifest


def available_jkp_years(data_dir):
    """Return the years present in the cleaned annual cache."""
    clean_dir, manifest = build_clean_jkp_cache(data_dir)
    years = []
    for filename in manifest["output_files"]:
        try:
            years.append(int(Path(filename).stem.rsplit("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    if not years:
        raise ValueError(f"No annual files found in cleaned cache: {clean_dir}")
    return sorted(set(years))


def available_jkp_characteristics(data_dir):
    """Return all characteristics retained by the cleaned JKP cache."""
    _, manifest = build_clean_jkp_cache(data_dir)
    return list(manifest["kept_characteristics"])


def load_jkp_value(data_dir, CHARACTERISTIC=None, years=None):
    """Load already-cleaned JKP USA data with selected characteristics.

    The cleaned cache is created automatically on the first invocation and is
    reused until one of the raw source files changes. If ``years`` is given,
    only those annual cache files are loaded.
    """
    clean_dir, manifest = build_clean_jkp_cache(data_dir)
    if isinstance(CHARACTERISTIC, str) and CHARACTERISTIC.lower() == "all":
        characteristics = list(manifest["kept_characteristics"])
    else:
        characteristics = _as_characteristic_list(CHARACTERISTIC)
    available = set(manifest["kept_characteristics"])
    unavailable = [column for column in characteristics if column not in available]
    if unavailable:
        raise ValueError(
            "Requested characteristics are not available after cleaning: "
            f"{unavailable}"
        )

    output_files = manifest["output_files"]
    if years is not None:
        requested_years = {int(year) for year in years}
        files_by_year = {
            int(Path(filename).stem.rsplit("_", 1)[1]): filename
            for filename in output_files
        }
        missing_years = sorted(requested_years - files_by_year.keys())
        if missing_years:
            raise FileNotFoundError(
                f"No cleaned JKP files found for years: {missing_years}"
            )
        output_files = [files_by_year[year] for year in sorted(requested_years)]

    columns = list(dict.fromkeys(_REQUIRED_COLUMNS + characteristics))
    frames = [
        pd.read_parquet(clean_dir / filename, columns=columns)
        for filename in output_files
    ]
    if not frames:
        raise ValueError(f"The cleaned JKP cache in {clean_dir.resolve()} is empty.")

    df = pd.concat(frames, ignore_index=True)
    df["eom"] = pd.to_datetime(df["eom"])
    df = df.sort_values(["eom", "id"], kind="stable").reset_index(drop=True)
    if characteristics and df[characteristics].isna().any().any():
        raise AssertionError("NaNs remain in the loaded clean characteristics.")
    if df["ret_exc_lead1m"].notna().mean() <= 0.5:
        raise AssertionError("Too many missing values in ret_exc_lead1m.")

    df.attrs["characteristics_rank_standardized"] = True
    df.attrs["characteristic_range"] = [-0.5, 0.5]
    df.attrs["clean_cache_dir"] = str(clean_dir.resolve())
    return df


def main():
    """Print a small preview when this file is executed directly."""
    import sys

    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from Utils.utils import build_monthly_panels

    characteristic = "be_me"
    df = load_jkp_value(project_root / "data" / "JKP_USA", characteristic)
    panels = build_monthly_panels(df, characteristics=characteristic)

    if not panels:
        raise ValueError("No valid monthly panels could be built from the dataset.")

    panel = panels[0]
    out = pd.DataFrame(
        {
            "id": panel["id"],
            "permno": panel["permno"],
            "r": panel["r"],
            f"{characteristic}_rank": panel["x"][:, 0],
        }
    )
    print("Date:", panel["date"])
    print(out.head(20).to_string())


if __name__ == "__main__":
    main()
