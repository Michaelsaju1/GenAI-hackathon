"""
Fetch North Korea military events from GDELT.

Primary: Google BigQuery (fast, requires gcloud auth)
Fallback: GDELT v1 daily CSV files (no auth required)
"""

import sys
import time
import zipfile
import io
import urllib.request
from datetime import datetime, timedelta

import pandas as pd

from config import (
    NK_ACTOR_CODE,
    CAMEO_MILITARY_ROOT_CODES,
    GDELT_BQ_TABLE,
    GDELT_FIELDS,
    GDELT_START_DATE,
    RAW_DIR,
)

OUTPUT_FILE = RAW_DIR / "nk_military_events.csv"

# GDELT v1 column indices (headerless, tab-separated, 58 columns)
COL_MAP = {
    0: "GLOBALEVENTID",
    1: "SQLDATE",
    6: "Actor1Name",
    7: "Actor1CountryCode",
    8: "Actor1Type1Code",
    16: "Actor2Name",
    17: "Actor2CountryCode",
    18: "Actor2Type1Code",
    26: "EventCode",
    27: "EventBaseCode",
    28: "EventRootCode",
    30: "GoldsteinScale",
    31: "NumMentions",
    32: "NumSources",
    33: "NumArticles",
    34: "AvgTone",
    36: "Actor1Geo_FullName",
    44: "Actor2Geo_FullName",
    51: "ActionGeo_FullName",
    53: "ActionGeo_Lat",
    54: "ActionGeo_Long",
    57: "SOURCEURL",
}
COL_INDICES = sorted(COL_MAP.keys())


def fetch_via_bigquery() -> pd.DataFrame:
    """Fetch GDELT events using Google BigQuery."""
    from google.cloud import bigquery

    client = bigquery.Client(project="project-db289aa8-2928-4a01-ba2")

    root_codes = ", ".join(f"'{c}'" for c in CAMEO_MILITARY_ROOT_CODES)
    fields = ", ".join(GDELT_FIELDS)

    query = f"""
    SELECT {fields}
    FROM `{GDELT_BQ_TABLE}`
    WHERE (Actor1CountryCode = '{NK_ACTOR_CODE}' OR Actor2CountryCode = '{NK_ACTOR_CODE}')
      AND EventRootCode IN ({root_codes})
      AND SQLDATE >= {GDELT_START_DATE}
    ORDER BY SQLDATE
    """

    print("Running BigQuery query...")
    df = client.query(query).to_dataframe()
    print(f"  Retrieved {len(df):,} events")
    return df


def fetch_via_daily_files() -> pd.DataFrame:
    """
    Download GDELT v1 daily export files and filter for NK military events.
    Files are at: http://data.gdeltproject.org/events/YYYYMMDD.export.CSV.zip
    """
    print("Downloading GDELT v1 daily export files...")

    base_url = "http://data.gdeltproject.org/events/"
    start = datetime.strptime(str(GDELT_START_DATE), "%Y%m%d")
    end = datetime.now()
    root_codes_set = set(CAMEO_MILITARY_ROOT_CODES)

    all_rows = []
    total_days = (end - start).days
    current = start
    day_num = 0
    consecutive_failures = 0

    while current <= end:
        day_num += 1
        date_str = current.strftime("%Y%m%d")
        url = f"{base_url}{date_str}.export.CSV.zip"

        if day_num % 30 == 1:
            print(f"\n  Processing {current.strftime('%Y-%m')} (day {day_num}/{total_days})...")

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as response:
                zip_data = response.read()
            consecutive_failures = 0
        except Exception:
            consecutive_failures += 1
            if consecutive_failures > 7:
                print(f"  WARNING: {consecutive_failures} consecutive failures at {date_str}, skipping ahead...")
                consecutive_failures = 0
            current += timedelta(days=1)
            continue

        try:
            zf = zipfile.ZipFile(io.BytesIO(zip_data))
            csv_name = zf.namelist()[0]
            raw_text = zf.read(csv_name).decode("utf-8", errors="replace")
        except Exception:
            current += timedelta(days=1)
            continue

        day_count = 0
        for line in raw_text.split("\n"):
            if not line.strip():
                continue
            fields = line.split("\t")
            if len(fields) < 58:
                continue

            actor1cc = fields[7].strip()
            actor2cc = fields[17].strip()
            root_code = fields[28].strip()

            if root_code not in root_codes_set:
                continue
            if actor1cc != NK_ACTOR_CODE and actor2cc != NK_ACTOR_CODE:
                continue

            row = {COL_MAP[i]: fields[i].strip() for i in COL_INDICES}
            all_rows.append(row)
            day_count += 1

        if day_count > 0:
            print(f"    {date_str}: {day_count} events", end="  ", flush=True)

        current += timedelta(days=1)

        # Rate limiting — be polite
        if day_num % 10 == 0:
            time.sleep(0.1)

    if not all_rows:
        print("\nERROR: No events found!")
        sys.exit(1)

    print(f"\n\n  Total: {len(all_rows):,} NK military events across {day_num} days")

    df = pd.DataFrame(all_rows)

    # Convert numeric columns
    for col in ["GoldsteinScale", "AvgTone", "ActionGeo_Lat", "ActionGeo_Long"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["NumMentions", "NumSources", "NumArticles"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    df = df.sort_values("SQLDATE").reset_index(drop=True)
    return df


def print_summary(df: pd.DataFrame):
    """Print dataset summary statistics."""
    from config import CAMEO_DESCRIPTIONS

    print(f"\n{'='*60}")
    print("GDELT NK Military Events Summary")
    print(f"{'='*60}")
    print(f"Total events: {len(df):,}")
    print(f"Date range: {df['SQLDATE'].min()} — {df['SQLDATE'].max()}")
    print(f"\nEvents by root code:")
    for code, count in df["EventRootCode"].value_counts().sort_index().items():
        desc = CAMEO_DESCRIPTIONS.get(str(code), "Unknown")
        print(f"  {code} ({desc}): {count:,}")
    print(f"\nGoldstein Scale: mean={df['GoldsteinScale'].mean():.2f}, "
          f"min={df['GoldsteinScale'].min():.1f}, max={df['GoldsteinScale'].max():.1f}")
    print(f"Avg Tone: mean={df['AvgTone'].mean():.2f}")
    print(f"\nSample events:")
    sample = df.sample(min(5, len(df)), random_state=42)
    for _, row in sample.iterrows():
        print(f"  [{row['SQLDATE']}] {row['Actor1Name']} -> {row['EventRootCode']} -> {row['Actor2Name']} "
              f"(Goldstein: {row['GoldsteinScale']}, Loc: {row['ActionGeo_FullName']})")


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Try BigQuery first, fall back to daily file downloads
    try:
        print("Attempting BigQuery access...")
        df = fetch_via_bigquery()
    except Exception as e:
        print(f"BigQuery failed: {e}")
        print("Falling back to GDELT v1 daily file download...\n")
        df = fetch_via_daily_files()

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved {len(df):,} events to {OUTPUT_FILE}")

    print_summary(df)


if __name__ == "__main__":
    main()
