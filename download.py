"""Resumable WRDS acquisition: bounded connections, year-sized SELECTs, no duplicate work."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import pandas as pd


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str))


def periods():
    for year in range(1963, 2025):
        yield str(year), pd.Timestamp(year=year, month=1, day=1), pd.Timestamp(year=year + 1, month=1, day=1)
    # January 2025 is downloaded only so a missing Dec-2024 lead return can be
    # reconciled from an actually observed next-calendar-month return.
    yield "2025-01", pd.Timestamp("2025-01-01"), pd.Timestamp("2025-02-01")


def download(out, names_file, max_connections=1):
    import psycopg2
    from psycopg2 import OperationalError

    names = json.loads(Path(names_file).read_text())
    if len(names) != 153 or len(set(names)) != 153:
        raise ValueError("Expected the 153 published JKP characteristic names.")
    if any(not n.replace("_", "").isalnum() for n in names):
        raise ValueError("Invalid column identifier.")
    if not 1 <= max_connections <= 5:
        raise ValueError("max_connections must be between one and five.")

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    manifest_file = out / "manifest.json"
    progress_file = out / "connection_attempt.json"

    if manifest_file.exists():
        manifest = json.loads(manifest_file.read_text())
        if manifest.get("status") != "complete":
            raise ValueError("Existing raw manifest is not complete.")
        for f in manifest["files"]:
            if sha256(out / f["name"]) != f["sha256"]:
                raise ValueError("Cached raw data checksum mismatch.")
        print("Existing complete raw cache reused. WRDS connections: 0.", flush=True)
        return

    username = os.environ.get("WRDS_USERNAME")
    password = os.environ.get("WRDS_PASSWORD")
    if not username or not password:
        raise RuntimeError("WRDS_USERNAME and WRDS_PASSWORD secrets are required.")

    columns = [
        "id", "eom", "excntry", "gvkey", "permno", "size_grp", "me",
        "ret_exc_lead1m", "common", "exch_main", "primary_sec", "obs_main",
        "crsp_shrcd", "crsp_exchcd", *names,
    ]
    select = ", ".join('g."' + n + '"' for n in columns)
    sql = (
        "SELECT " + select +
        ", g.ret_exc AS current_excess_return, g.ret AS current_total_return "
        "FROM contrib.global_factor g "
        "WHERE excntry = 'USA' AND id <= 99999 "
        "AND eom >= %s AND eom < %s ORDER BY eom, id"
    )
    (out / "query_template.sql").write_text(sql)

    expected = list(periods())
    completed = {}

    state = {
        "status": "starting",
        "connection_attempts": 0,
        "select_calls": 0,
        "completed_periods": [],
        "current_period": None,
        "phase": None,
        "last_error_type": None,
        "last_error_message": None,
    }
    if progress_file.exists():
        old = json.loads(progress_file.read_text())
        for item in old.get("saved_files", []):
            p = out / item["name"]
            if p.exists() and sha256(p) == item["sha256"]:
                completed[item["period"]] = p
    started = time.monotonic()

    files = []
    saved_by_period = {}
    for label, p in completed.items():
        saved_by_period[label] = {
            "period": label, "name": p.name, "rows": int(pd.read_parquet(p, columns=["id"]).shape[0]),
            "sha256": sha256(p),
        }

    def persist():
        state["saved_files"] = [saved_by_period[k] for k in sorted(saved_by_period)]
        state["completed_periods"] = sorted(saved_by_period)
        write_json(progress_file, state)

    def convert(frame):
        frame["eom"] = pd.to_datetime(frame["eom"])
        for col in names + ["me", "ret_exc_lead1m", "current_excess_return", "current_total_return"]:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").astype(float)
        for col in [
            "id", "permno", "crsp_shrcd", "crsp_exchcd",
            "common", "exch_main", "primary_sec", "obs_main",
        ]:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("Int64")
        for col in ["excntry", "gvkey", "size_grp"]:
            frame[col] = frame[col].astype("string")
        return frame

    remaining = [(a, b, c) for a, b, c in expected if a not in saved_by_period]
    while remaining and state["connection_attempts"] < max_connections:
        state["connection_attempts"] += 1
        state["status"] = "connecting"
        state["phase"] = "connect"
        persist()
        conn = None
        try:
            print(
                f"WRDS connection {state['connection_attempts']}/{max_connections}; "
                f"{len(remaining)} periods remain.",
                flush=True,
            )
            conn = psycopg2.connect(
                user=username,
                password=password,
                host="wrds-pgdata.wharton.upenn.edu",
                port=9737,
                dbname="wrds",
                sslmode="require",
                connect_timeout=30,
                application_name="portfolio_paper_yearly_download",
                keepalives=1,
                keepalives_idle=30,
                keepalives_interval=10,
                keepalives_count=5,
            )
            conn.set_session(readonly=True, autocommit=False)
            for label, start, end in list(remaining):
                state["status"] = "querying"
                state["current_period"] = label
                state["phase"] = "execute"
                state["select_calls"] += 1
                persist()
                print(f"WRDS SELECT {label}: {start.date()} to {end.date()}.", flush=True)
                buffers = []
                cursor = conn.cursor(name="paper_" + label.replace("-", "_"))
                cursor.itersize = 5000
                try:
                    cursor.execute(sql, (start.date(), end.date()))
                    state["phase"] = "fetch"
                    persist()
                    while True:
                        rows = cursor.fetchmany(5000)
                        if not rows:
                            break
                        buffers.append(pd.DataFrame.from_records(
                            rows, columns=columns + ["current_excess_return", "current_total_return"]
                        ))
                finally:
                    cursor.close()
                if not buffers:
                    raise RuntimeError(f"WRDS returned no rows for {label}.")
                frame = convert(pd.concat(buffers, ignore_index=True))
                path = out / f"jkp_{label}.parquet"
                frame.to_parquet(path, index=False, compression="zstd")
                item = {
                    "period": label,
                    "name": path.name,
                    "rows": len(frame),
                    "sha256": sha256(path),
                }
                saved_by_period[label] = item
                state["phase"] = "saved"
                state["last_error_type"] = None
                state["last_error_message"] = None
                persist()
                conn.commit()
                remaining = [(a, b, c) for a, b, c in remaining if a != label]
                print(f"Raw {label}: {len(frame):,} rows saved.", flush=True)
        except OperationalError as error:
            state["status"] = "connection_lost"
            state["last_error_type"] = type(error).__name__
            state["last_error_message"] = "Connection failure; details suppressed to protect credentials"
            persist()
            print(
                f"WRDS connection lost during {state['current_period']} / {state['phase']}; "
                f"saved periods will not be repeated.",
                flush=True,
            )
        except Exception:
            state["status"] = "failed"
            state["last_error_message"] = "Acquisition failed; details suppressed to protect credentials"
            persist()
            raise RuntimeError("WRDS acquisition failed; no database exception text is exposed.") from None
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

    if remaining:
        state["status"] = "failed"
        state["remaining_periods"] = [x[0] for x in remaining]
        persist()
        raise RuntimeError(
            f"WRDS acquisition stopped after {state['connection_attempts']} bounded "
            f"connection attempts; {len(remaining)} periods remain."
        )

    files = [saved_by_period[label] for label, _, _ in expected]
    total = sum(item["rows"] for item in files)
    manifest = {
        "status": "complete",
        "source": "WRDS contrib.global_factor",
        "return_units": "decimal",
        "return_construction": "JKP ret_exc and ret_exc_lead1m, including upstream CRSP delisting treatment",
        "acquired_at":str(pd.Timestamp.now("UTC")),
        "characteristics": names,
        "files": [{k: v for k, v in item.items() if k != "period"} for item in files],
        "periods": [item["period"] for item in files],
        "rows": total,
        "git_sha": os.environ.get("GITHUB_SHA"),
        "elapsed_seconds": time.monotonic() - started,
        "connection_attempts": state["connection_attempts"],
        "select_calls": state["select_calls"],
        "query_template_sha256": sha256(out / "query_template.sql"),
        "download_design": "year-sized SELECTs on bounded persistent connections",
    }
    write_json(manifest_file, manifest)
    state["status"] = "complete"
    state["current_period"] = None
    state["phase"] = "complete"
    persist()
    print(
        f"WRDS acquisition complete: {total:,} rows, "
        f"{state['select_calls']} SELECTs, {state['connection_attempts']} connections.",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/raw"))
    parser.add_argument("--names", type=Path, required=True)
    parser.add_argument("--max-connections", type=int, default=1)
    args = parser.parse_args()
    download(args.out, args.names, args.max_connections)
