"""Download the JKP U.S. stock-level sample with exactly one WRDS request.

The downloader intentionally opens one WRDS connection and executes one SQL
statement for the entire 1963--2024 sample. Results are streamed in chunks and
split into annual parquet files locally.

This matters operationally: one run can trigger at most one external WRDS/CRSP
authentication attempt (and therefore at most one Duo request).
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import sqlalchemy as sa


START_YEAR = 1963
END_YEAR = 2024
MAX_WRDS_REQUESTS = 1
CHUNK_SIZE = 100_000


class SingleRequestWRDS:
    """WRDS client with a hard one-request budget.

    A request means one connection attempt plus one SQL statement. There are
    deliberately no automatic retries: if authentication or the query fails,
    the run fails rather than triggering another Duo/WRDS request.
    """

    def __init__(self, engine, max_requests=MAX_WRDS_REQUESTS):
        self.engine = engine
        self.max_requests = int(max_requests)
        self.request_count = 0
        self.connection_attempts = 0

    def stream_sql(self, query, *, date_cols=None, chunksize=CHUNK_SIZE):
        if self.request_count >= self.max_requests:
            raise RuntimeError(
                "WRDS request budget exhausted: this run is allowed at most "
                f"{self.max_requests} external WRDS request."
            )

        self.request_count += 1
        self.connection_attempts += 1

        # Exactly one connection attempt. No retry loop and no pool pre-ping.
        with self.engine.connect() as raw_connection:
            connection = raw_connection.execution_options(
                stream_results=True,
                max_row_buffer=chunksize,
            )
            chunks = pd.read_sql_query(
                sa.text(query),
                connection,
                parse_dates=date_cols,
                chunksize=chunksize,
            )
            for chunk in chunks:
                yield chunk

    def assert_single_request(self):
        if self.request_count != 1:
            raise AssertionError(
                f"Expected exactly one WRDS SQL request, observed {self.request_count}."
            )
        if self.connection_attempts != 1:
            raise AssertionError(
                "Expected exactly one WRDS connection attempt, observed "
                f"{self.connection_attempts}."
            )

    def close(self):
        self.engine.dispose()


def build_engine(username, password):
    """Build a no-retry, no-pool WRDS engine."""
    return sa.create_engine(
        sa.URL.create(
            drivername="postgresql+psycopg2",
            username=username,
            password=password,
            host="wrds-pgdata.wharton.upenn.edu",
            port=9737,
            database="wrds",
        ),
        poolclass=sa.pool.NullPool,
        connect_args={
            "sslmode": "require",
            "application_name": "portfolio_learnability_single_request",
            "connect_timeout": 30,
            "keepalives": 1,
            "keepalives_idle": 30,
            "keepalives_interval": 10,
            "keepalives_count": 5,
        },
    )


def official_jkp_characteristics():
    """Read the public JKP characteristic list (not a WRDS request)."""
    factor_details_url = (
        "https://raw.githubusercontent.com/"
        "bkelly-lab/jkp-data/main/"
        "src/jkp/data/resources/factor_details.xlsx"
    )
    factor_details = pd.read_excel(factor_details_url)
    return (
        factor_details.loc[
            factor_details["abr_jkp"].notna(),
            "abr_jkp",
        ]
        .astype(str)
        .drop_duplicates()
        .tolist()
    )


def _flush_year(output_dir, year, parts):
    if year is None or not parts:
        return None

    frame = pd.concat(parts, ignore_index=True)
    frame = frame.sort_values(["eom", "id"], kind="stable").reset_index(drop=True)

    filename = output_dir / f"JKP_USA_{year}.parquet"
    temporary = filename.with_suffix(".parquet.tmp")
    frame.to_parquet(
        temporary,
        index=False,
        compression="zstd",
    )
    temporary.replace(filename)

    print(
        f"{year}: {len(frame):,} stock-months -> {filename}",
        flush=True,
    )
    return filename


def download_sample(db, chars, output_dir):
    """Execute the single WRDS query and split the streamed result locally."""
    base_cols = [
        "id",
        "eom",
        "excntry",
        "gvkey",
        "permno",
        "size_grp",
        "me",
        "ret_exc_lead1m",
    ]
    columns_sql = ", ".join(base_cols + chars)

    filters = """
        common = 1
        AND exch_main = 1
        AND primary_sec = 1
        AND obs_main = 1
        AND excntry = 'USA'
    """

    query = f"""
        SELECT
            {columns_sql}
        FROM contrib.global_factor
        WHERE
            {filters}
            AND eom >= '{START_YEAR}-01-01'
            AND eom < '{END_YEAR + 1}-01-01'
        ORDER BY eom, id
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    current_year = None
    current_parts = []
    written = []

    for chunk in db.stream_sql(
        query,
        date_cols=["eom"],
        chunksize=CHUNK_SIZE,
    ):
        if chunk.empty:
            continue

        chunk["eom"] = pd.to_datetime(chunk["eom"])
        years = chunk["eom"].dt.year

        for year, part in chunk.groupby(years, sort=True):
            year = int(year)
            if year < START_YEAR or year > END_YEAR:
                continue

            if current_year is None:
                current_year = year

            if year != current_year:
                saved = _flush_year(
                    output_dir,
                    current_year,
                    current_parts,
                )
                if saved is not None:
                    written.append(saved)
                current_year = year
                current_parts = []

            current_parts.append(part.copy())

    saved = _flush_year(
        output_dir,
        current_year,
        current_parts,
    )
    if saved is not None:
        written.append(saved)

    db.assert_single_request()

    years_written = sorted(
        int(path.stem.rsplit("_", 1)[1])
        for path in written
    )
    if not years_written:
        raise RuntimeError("The single WRDS query returned no observations.")
    if years_written[0] != START_YEAR or years_written[-1] != END_YEAR:
        raise RuntimeError(
            "Unexpected sample bounds from WRDS: "
            f"{years_written[0]}--{years_written[-1]}."
        )

    return written


def main():
    username = os.environ.get("WRDS_USERNAME")
    password = os.environ.get("WRDS_PASSWORD")

    if not username or not password:
        raise RuntimeError(
            "WRDS credentials are required through WRDS_USERNAME and "
            "WRDS_PASSWORD environment variables."
        )

    chars = official_jkp_characteristics()
    print(f"Number of JKP characteristics: {len(chars)}", flush=True)
    print(
        "WRDS request budget: exactly one connection / one SQL statement. "
        "Automatic retries are disabled.",
        flush=True,
    )

    engine = build_engine(username, password)
    db = SingleRequestWRDS(engine)

    try:
        written = download_sample(
            db,
            chars,
            Path("data/JKP_USA"),
        )
    finally:
        db.close()

    print(
        f"DONE: {len(written)} annual files; "
        f"WRDS requests={db.request_count}; "
        f"connection attempts={db.connection_attempts}",
        flush=True,
    )


if __name__ == "__main__":
    main()
