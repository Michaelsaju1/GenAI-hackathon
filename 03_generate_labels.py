"""
Generate risk assessment labels for each weekly cluster using Claude API.

Input:  data/clusters/all_clusters.jsonl
Output: data/labeled/all_labeled.jsonl

Supports resuming — skips clusters that already have labels.
"""

import json
import time
import sys

from dotenv import load_dotenv
load_dotenv()

import anthropic

from config import (
    CLUSTERS_DIR,
    LABELED_DIR,
    LABEL_MODEL,
    LABEL_TEMPERATURE,
    RISK_ASSESSMENT_FIELDS,
)

INPUT_FILE = CLUSTERS_DIR / "all_clusters.jsonl"
OUTPUT_FILE = LABELED_DIR / "all_labeled.jsonl"

SYSTEM_PROMPT = """You are an expert geopolitical risk analyst specializing in North Korean military affairs and Korean Peninsula security dynamics. You have deep knowledge of DPRK military doctrine, the history of inter-Korean tensions, US-ROK alliance dynamics, and UN sanctions enforcement.

Given a weekly summary of events from the GDELT database (which codes worldwide news into structured events), produce a structured risk assessment in JSON format.

Your assessment must be grounded in the data provided. Do not fabricate specific events, troop numbers, or weapon details not reflected in the data. When data is sparse, say so."""

USER_TEMPLATE = """Analyze the following GDELT event data for North Korean military activity and produce a structured risk assessment.

{summary_text}

Respond with ONLY a valid JSON object containing these fields:
- "escalation_level": integer 1-5 (1=routine posturing/normal diplomatic friction, 2=elevated rhetoric or increased activity, 3=provocative military actions or demonstrations, 4=significant military escalation or crisis, 5=active armed conflict)
- "escalation_rationale": 2-3 sentences explaining the escalation rating based on the data
- "situation_summary": 3-5 sentence assessment of the week's developments
- "key_actors": list of objects with "name", "role", and "posture" fields
- "historical_context": 1-2 sentences connecting these events to broader patterns
- "watch_indicators": list of 3-5 specific things to monitor going forward
- "potential_trajectories": list of 2-3 possible near-term scenarios
- "confidence_level": "low", "medium", or "high" based on data volume and source diversity
- "data_caveats": 1-2 sentences on limitations of this assessment"""


def load_existing_labels() -> set:
    """Load already-labeled week_starts to support resuming."""
    labeled = set()
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE) as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    labeled.add(obj["week_start"])
    return labeled


def generate_label(client: anthropic.Anthropic, cluster: dict) -> dict | None:
    """Generate a risk assessment for one cluster."""
    try:
        response = client.messages.create(
            model=LABEL_MODEL,
            max_tokens=1500,
            temperature=LABEL_TEMPERATURE,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": USER_TEMPLATE.format(summary_text=cluster["summary_text"]),
                }
            ],
        )

        text = response.content[0].text.strip()

        # Handle potential markdown code fences
        if text.startswith("```"):
            text = text.split("\n", 1)[1]  # Remove first line
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        assessment = json.loads(text)

        # Validate required fields
        missing = [f for f in RISK_ASSESSMENT_FIELDS if f not in assessment]
        if missing:
            print(f"    WARNING: Missing fields: {missing}")

        return assessment

    except json.JSONDecodeError as e:
        print(f"    ERROR: Invalid JSON response: {e}")
        print(f"    Raw text: {text[:200]}...")
        return None
    except Exception as e:
        print(f"    ERROR: API call failed: {e}")
        return None


def main():
    LABELED_DIR.mkdir(parents=True, exist_ok=True)

    # Load clusters
    clusters = []
    with open(INPUT_FILE) as f:
        for line in f:
            if line.strip():
                clusters.append(json.loads(line))
    print(f"Loaded {len(clusters)} clusters from {INPUT_FILE}")

    # Check for existing progress
    already_labeled = load_existing_labels()
    to_process = [c for c in clusters if c["week_start"] not in already_labeled]
    print(f"Already labeled: {len(already_labeled)}, remaining: {len(to_process)}")

    if not to_process:
        print("All clusters already labeled. Done!")
        return

    # Initialize Anthropic client
    client = anthropic.Anthropic()  # Uses ANTHROPIC_API_KEY env var

    # Process clusters
    success_count = 0
    fail_count = 0

    with open(OUTPUT_FILE, "a") as f:
        for i, cluster in enumerate(to_process):
            print(f"  [{i+1}/{len(to_process)}] {cluster['week_start']} "
                  f"({cluster['num_events']} events)...", end=" ", flush=True)

            assessment = generate_label(client, cluster)

            if assessment is None:
                # Retry once
                print("Retrying...", end=" ", flush=True)
                time.sleep(2)
                assessment = generate_label(client, cluster)

            if assessment is not None:
                labeled = {
                    "week_start": cluster["week_start"],
                    "week_end": cluster["week_end"],
                    "num_events": cluster["num_events"],
                    "summary_text": cluster["summary_text"],
                    "risk_assessment": assessment,
                }
                f.write(json.dumps(labeled) + "\n")
                f.flush()
                success_count += 1
                esc = assessment.get("escalation_level", "?")
                conf = assessment.get("confidence_level", "?")
                print(f"OK escalation={esc} confidence={conf}")
            else:
                fail_count += 1
                print("FAILED")

            # Rate limiting
            time.sleep(0.5)

    print(f"\nDone! Success: {success_count}, Failed: {fail_count}")
    print(f"Total labeled: {len(already_labeled) + success_count}")

    # Print escalation distribution
    all_labeled = []
    with open(OUTPUT_FILE) as f:
        for line in f:
            if line.strip():
                all_labeled.append(json.loads(line))

    levels = [l["risk_assessment"]["escalation_level"] for l in all_labeled
              if "escalation_level" in l.get("risk_assessment", {})]
    if levels:
        print(f"\nEscalation level distribution:")
        for level in sorted(set(levels)):
            count = levels.count(level)
            print(f"  Level {level}: {count} ({count/len(levels)*100:.1f}%)")


if __name__ == "__main__":
    main()
