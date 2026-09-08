import wrds
import pandas as pd
from pathlib import Path


# ============================================================
# 1. WRDS CREDENTIALS
# ============================================================

WRDS_USERNAME = "bignolo1111"
WRDS_PASSWORD = "YZag6ycw7f9GKGt"


# ============================================================
# 2. CONNECT TO WRDS -- NO PROMPT
# ============================================================

db = wrds.Connection(
    wrds_username=WRDS_USERNAME,
    wrds_password=WRDS_PASSWORD,
)


# ============================================================
# 3. GET OFFICIAL JKP CHARACTERISTICS
# ============================================================

factor_details_url = (
    "https://raw.githubusercontent.com/"
    "bkelly-lab/jkp-data/main/"
    "src/jkp/data/resources/factor_details.xlsx"
)

factor_details = pd.read_excel(factor_details_url)

chars = (
    factor_details
    .loc[
        factor_details["abr_jkp"].notna(),
        "abr_jkp"
    ]
    .astype(str)
    .drop_duplicates()
    .tolist()
)

print(f"Number of JKP characteristics: {len(chars)}")


# ============================================================
# 4. COLUMNS
# ============================================================

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

columns = base_cols + chars
columns_sql = ", ".join(columns)


# ============================================================
# 5. JKP USA FILTERS
# ============================================================

filters = """
    common = 1
    AND exch_main = 1
    AND primary_sec = 1
    AND obs_main = 1
    AND excntry = 'USA'
"""


# ============================================================
# 6. AVAILABLE DATE RANGE
# ============================================================

bounds = db.raw_sql(
    f"""
    SELECT
        MIN(eom) AS min_date,
        MAX(eom) AS max_date
    FROM contrib.global_factor
    WHERE {filters}
    """,
    date_cols=["min_date", "max_date"],
)

start_year = pd.Timestamp(bounds.loc[0, "min_date"]).year
end_year = pd.Timestamp(bounds.loc[0, "max_date"]).year

print("JKP USA sample:")
print(bounds)


# ============================================================
# 7. OUTPUT DIRECTORY
# ============================================================

output_dir = Path("data/JKP_USA")
output_dir.mkdir(parents=True, exist_ok=True)


# ============================================================
# 8. DOWNLOAD YEAR BY YEAR
# ============================================================

for year in range(start_year, end_year + 1):

    filename = output_dir / f"JKP_USA_{year}.parquet"

    if filename.exists():
        print(f"{year}: already exists -> skipping")
        continue

    print(f"{year}: downloading...")

    sql = f"""
        SELECT
            {columns_sql}

        FROM contrib.global_factor

        WHERE
            {filters}

            AND eom >= '{year}-01-01'
            AND eom < '{year + 1}-01-01'

        ORDER BY
            eom,
            id
    """

    df = db.raw_sql(
        sql,
        date_cols=["eom"],
    )

    if df.empty:
        print(f"{year}: no observations")
        continue

    df.to_parquet(
        filename,
        index=False,
        compression="zstd",
    )

    print(
        f"{year}: "
        f"{len(df):,} stock-months -> "
        f"{filename}"
    )


# ============================================================
# 9. CLOSE
# ============================================================

db.close()

print("\nDONE")