import re
from pathlib import Path

import pandas as pd


INPUT_CSV = Path("earnings_calls.csv")
OUTPUT_OCCURRENCES = Path("dynamic_capability_keyword_occurrences.csv")
OUTPUT_SEGMENTS = Path("dynamic_capability_wtt_segments.csv")
OUTPUT_COUNTS = Path("dynamic_capability_keyword_call_counts.csv")

WINDOW_WORDS = 4

KEYWORD_GROUPS = {
    "sensing": [
        "sense",
        "senses",
        "sensed",
        "sensing",
        "shape",
        "shaped",
        "shaping",
    ],
    "seizing": [
        "seize",
        "seized",
        "seizing",
        "invest",
        "investing",
        "invested",
    ],
    "transforming": [
        "transform",
        "transforming",
        "transformed",
        "reconfigure",
        "reconfigured",
        "reconfiguring",
        "redeployment",
        "redeploy",
        "redeployed",
        "redesign",
        "redesigned",
        "redesigning",
        "revamping",
        "revamped",
        "revamp",
    ],
}


def clean_date_column(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype("string")
        .str.replace("a.m.", "AM", regex=False)
        .str.replace("p.m.", "PM", regex=False)
        .str.replace(r"\s+ET$", "", regex=True)
    )
    return pd.to_datetime(cleaned, format="mixed", errors="coerce")


def tokenize(text: str) -> list[str]:
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.findall(r"[a-z]+", text)


def merge_windows(windows: list[dict]) -> list[dict]:
    if not windows:
        return []

    sorted_windows = sorted(windows, key=lambda item: (item["start"], item["end"]))
    merged = [sorted_windows[0].copy()]

    for window in sorted_windows[1:]:
        current = merged[-1]
        if window["start"] <= current["end"]:
            current["end"] = max(current["end"], window["end"])
            current["matched_terms"].extend(window["matched_terms"])
        else:
            merged.append(window.copy())

    for window in merged:
        window["matched_terms"] = sorted(set(window["matched_terms"]))

    return merged


def build_keyword_dataframes(calls: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    keyword_to_group = {
        keyword: group
        for group, keywords in KEYWORD_GROUPS.items()
        for keyword in keywords
    }
    keyword_set = set(keyword_to_group)

    occurrence_rows = []
    segment_rows = []
    count_rows = []

    metadata_cols = ["date", "exchange", "q", "ticker"]

    for call_id, row in calls.iterrows():
        tokens = tokenize(row["transcript"])
        word_count = len(tokens)
        group_counts = {group: 0 for group in KEYWORD_GROUPS}
        keyword_counts = {keyword: 0 for keyword in keyword_to_group}
        windows_by_group = {group: [] for group in KEYWORD_GROUPS}

        base = {
            "call_id": call_id,
            **{col: row.get(col) for col in metadata_cols},
        }

        for token_idx, token in enumerate(tokens):
            if token not in keyword_set:
                continue

            group = keyword_to_group[token]
            group_counts[group] += 1
            keyword_counts[token] += 1

            start = max(0, token_idx - WINDOW_WORDS)
            end = min(len(tokens), token_idx + WINDOW_WORDS + 1)
            left_context = " ".join(tokens[start:token_idx])
            right_context = " ".join(tokens[token_idx + 1:end])
            kwic_text = " ".join(tokens[start:end])

            occurrence_rows.append(
                {
                    **base,
                    "capability": group,
                    "keyword": token,
                    "matched_term": token,
                    "match_token_index": token_idx,
                    "window_start": start,
                    "window_end": end,
                    "left_context": left_context,
                    "right_context": right_context,
                    "kwic_text": kwic_text,
                }
            )

            windows_by_group[group].append(
                {
                    "start": start,
                    "end": end,
                    "matched_terms": [token],
                }
            )

        for group, windows in windows_by_group.items():
            for segment_id, window in enumerate(merge_windows(windows), start=1):
                segment_rows.append(
                    {
                        **base,
                        "capability": group,
                        "segment_id": segment_id,
                        "window_start": window["start"],
                        "window_end": window["end"],
                        "matched_terms": ", ".join(window["matched_terms"]),
                        "segment_text": " ".join(tokens[window["start"]:window["end"]]),
                    }
                )

        total_count = sum(group_counts.values())
        count_row = {
            **base,
            "word_count": word_count,
            "total_dynamic_capability_keywords": total_count,
        }
        for group, count in group_counts.items():
            count_row[f"{group}_count"] = count
            count_row[f"{group}_per_1000_words"] = (count / word_count * 1000) if word_count else 0
        count_row["total_dynamic_capability_keywords_per_1000_words"] = (
            total_count / word_count * 1000
        ) if word_count else 0
        for keyword, count in keyword_counts.items():
            count_row[f"keyword_{keyword}_count"] = count

        count_rows.append(count_row)

    occurrences = pd.DataFrame(occurrence_rows)
    segments = pd.DataFrame(segment_rows)
    counts = pd.DataFrame(count_rows)
    return occurrences, segments, counts


def main() -> None:
    calls = pd.read_csv(INPUT_CSV)
    calls["date"] = clean_date_column(calls["date"])
    calls["exchange"] = calls["exchange"].astype("string").str.split(":").str[0]

    occurrences, segments, counts = build_keyword_dataframes(calls)

    occurrences.to_csv(OUTPUT_OCCURRENCES, index=False)
    segments.to_csv(OUTPUT_SEGMENTS, index=False)
    counts.to_csv(OUTPUT_COUNTS, index=False)

    print(f"Saved {len(occurrences):,} keyword occurrences to {OUTPUT_OCCURRENCES}")
    print(f"Saved {len(segments):,} WTT-style merged text segments to {OUTPUT_SEGMENTS}")
    print(f"Saved {len(counts):,} call-level count rows to {OUTPUT_COUNTS}")
    print()
    print("Category totals:")
    print(counts[["sensing_count", "seizing_count", "transforming_count"]].sum().to_string())


if __name__ == "__main__":
    main()
