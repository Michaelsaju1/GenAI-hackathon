"""
Group NK military events into weekly clusters and produce structured summaries.

Input:  data/raw/nk_military_events.csv
Output: data/clusters/all_clusters.jsonl
"""

import json
import pandas as pd

from config import (
    RAW_DIR,
    CLUSTERS_DIR,
    CAMEO_DESCRIPTIONS,
    MIN_EVENTS_PER_CLUSTER,
    CLUSTER_FREQ,
)

INPUT_FILE = RAW_DIR / "nk_military_events.csv"
OUTPUT_FILE = CLUSTERS_DIR / "all_clusters.jsonl"


def build_cluster_summary(week_start: str, week_end: str, group: pd.DataFrame) -> dict:
    """Build a structured summary for one weekly cluster."""

    # Event breakdown by root code
    code_counts = group["EventRootCode"].value_counts().to_dict()
    event_breakdown = ", ".join(
        f"{CAMEO_DESCRIPTIONS.get(str(code), code)}({count})"
        for code, count in sorted(code_counts.items(), key=lambda x: -x[1])
    )

    # Unique actors
    actors1 = group["Actor1Name"].dropna().replace("", pd.NA).dropna().unique().tolist()
    actors2 = group["Actor2Name"].dropna().replace("", pd.NA).dropna().unique().tolist()
    all_actors = sorted(set(actors1 + actors2))[:10]  # Top 10

    # Unique locations
    locations = (
        group["ActionGeo_FullName"]
        .dropna()
        .replace("", pd.NA)
        .dropna()
        .unique()
        .tolist()
    )[:8]

    # Top events by mentions
    top_events = (
        group.nlargest(5, "NumMentions")[
            ["SQLDATE", "Actor1Name", "EventRootCode", "Actor2Name",
             "GoldsteinScale", "NumMentions", "ActionGeo_FullName"]
        ]
        .to_dict("records")
    )

    notable_lines = []
    for e in top_events:
        root = str(e["EventRootCode"])
        desc = CAMEO_DESCRIPTIONS.get(root, root)
        a1 = e["Actor1Name"] or "Unknown"
        a2 = e["Actor2Name"] or "Unknown"
        loc = e["ActionGeo_FullName"] or "Unknown location"
        notable_lines.append(
            f"{a1} ->{desc} ->{a2} at {loc} "
            f"(Goldstein: {e['GoldsteinScale']}, Mentions: {e['NumMentions']})"
        )

    # Top source URLs
    source_urls = (
        group.nlargest(3, "NumMentions")["SOURCEURL"]
        .dropna()
        .replace("", pd.NA)
        .dropna()
        .tolist()
    )

    summary_text = (
        f"Week: {week_start} to {week_end}\n"
        f"Total Events: {len(group)}\n"
        f"Event Breakdown: {event_breakdown}\n"
        f"Avg Goldstein Scale: {group['GoldsteinScale'].mean():.1f}\n"
        f"Avg Tone: {group['AvgTone'].mean():.1f}\n"
        f"Total Mentions: {group['NumMentions'].sum():,}\n"
        f"Total Sources: {group['NumSources'].sum():,}\n"
        f"Key Actors: {', '.join(all_actors) if all_actors else 'N/A'}\n"
        f"Locations: {', '.join(locations) if locations else 'N/A'}\n"
        f"Notable Events:\n"
        + "\n".join(f"  - {line}" for line in notable_lines)
    )

    return {
        "week_start": week_start,
        "week_end": week_end,
        "num_events": len(group),
        "summary_text": summary_text,
        "avg_goldstein": round(group["GoldsteinScale"].mean(), 2),
        "avg_tone": round(group["AvgTone"].mean(), 2),
        "total_mentions": int(group["NumMentions"].sum()),
        "event_breakdown": code_counts,
        "actors": all_actors,
        "locations": locations,
        "source_urls": source_urls,
    }


def main():
    CLUSTERS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading events from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE, dtype={"SQLDATE": str, "EventRootCode": str, "EventCode": str})
    print(f"  Loaded {len(df):,} events")

    # Parse dates
    df["date"] = pd.to_datetime(df["SQLDATE"], format="%Y%m%d")
    df = df.set_index("date").sort_index()

    # Group by week
    clusters = []
    for week_start, group in df.groupby(pd.Grouper(freq=CLUSTER_FREQ)):
        if len(group) < MIN_EVENTS_PER_CLUSTER:
            continue

        week_end = week_start + pd.Timedelta(days=6)
        cluster = build_cluster_summary(
            week_start.strftime("%Y-%m-%d"),
            week_end.strftime("%Y-%m-%d"),
            group,
        )
        clusters.append(cluster)

    print(f"\nBuilt {len(clusters)} weekly clusters (dropped weeks with < {MIN_EVENTS_PER_CLUSTER} events)")

    # Save
    with open(OUTPUT_FILE, "w") as f:
        for cluster in clusters:
            f.write(json.dumps(cluster) + "\n")
    print(f"Saved to {OUTPUT_FILE}")

    # Summary stats
    event_counts = [c["num_events"] for c in clusters]
    print(f"\nCluster statistics:")
    print(f"  Total clusters: {len(clusters)}")
    print(f"  Date range: {clusters[0]['week_start']} — {clusters[-1]['week_end']}")
    print(f"  Events per cluster: min={min(event_counts)}, max={max(event_counts)}, "
          f"mean={sum(event_counts)/len(event_counts):.1f}")

    # Print a sample cluster
    print(f"\n{'='*60}")
    print("Sample cluster:")
    print(f"{'='*60}")
    print(clusters[len(clusters)//2]["summary_text"])


if __name__ == "__main__":
    main()
