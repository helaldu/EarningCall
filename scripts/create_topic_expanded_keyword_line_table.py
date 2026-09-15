import re
from pathlib import Path

import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


CALLS_CSV = Path("earnings_calls.csv")
SEGMENTS_CSV = Path("dynamic_capability_wtt_segments.csv")
OUTPUT_TABLE = Path("dynamic_capability_topic_keyword_line_table.csv")
OUTPUT_TERMS = Path("dynamic_capability_topic_expanded_terms.csv")

TOP_WORDS_PER_TOPIC = 12
MIN_TERM_LENGTH = 4

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

# BERTopic surfaces some related words that are analytically useful, but it also
# surfaces broad earnings-call terms. Keep the final line table conservative.
CURATED_TOPIC_TERMS_FOR_LINE_TABLE = {
    "sensing": set(),
    "seizing": {"acquisition", "acquisitions", "capex"},
    "transforming": set(),
}

CUSTOM_STOPWORDS = {
    "able",
    "actually",
    "ahead",
    "analyst",
    "answer",
    "asked",
    "business",
    "call",
    "calls",
    "company",
    "conference",
    "continue",
    "continued",
    "continues",
    "doing",
    "earnings",
    "going",
    "good",
    "great",
    "growth",
    "like",
    "look",
    "looking",
    "maybe",
    "mean",
    "need",
    "operator",
    "people",
    "quarter",
    "question",
    "really",
    "right",
    "said",
    "thanks",
    "think",
    "time",
    "today",
    "want",
    "year",
    "years",
}


def clean_date_column(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype("string")
        .str.replace("a.m.", "AM", regex=False)
        .str.replace("p.m.", "PM", regex=False)
        .str.replace(r"\s+ET$", "", regex=True)
    )
    return pd.to_datetime(cleaned, format="mixed", errors="coerce")


def normalize_exchange(series: pd.Series) -> pd.Series:
    return series.astype("string").str.split(":").str[0]


def normalize_term(term: str) -> str:
    return re.sub(r"[^a-z\s]", " ", str(term).lower()).strip()


def is_useful_topic_word(term: str, seed_terms: set[str]) -> bool:
    term = normalize_term(term)
    if not term or " " in term:
        return False
    if len(term) < MIN_TERM_LENGTH:
        return False
    if term in ENGLISH_STOP_WORDS or term in CUSTOM_STOPWORDS:
        return False
    if term.isdigit():
        return False
    # Keep original seed terms even if they are short/common.
    if term in seed_terms:
        return True
    return True


def build_topic_expanded_terms(segments: pd.DataFrame) -> pd.DataFrame:
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    rows = []

    for category, seed_terms in KEYWORD_GROUPS.items():
        category_segments = (
            segments.loc[segments["capability"] == category, "segment_text"]
            .dropna()
            .astype(str)
            .tolist()
        )
        if not category_segments:
            continue

        model = BERTopic(
            embedding_model=embedding_model,
            min_topic_size=max(5, min(15, len(category_segments) // 20)),
            calculate_probabilities=False,
            verbose=False,
        )
        topics, _ = model.fit_transform(category_segments)

        seed_set = {normalize_term(term) for term in seed_terms}
        for term in seed_terms:
            rows.append(
                {
                    "category": category,
                    "term": normalize_term(term),
                    "source": "seed_keyword",
                    "topic": None,
                    "weight": None,
                    "used_in_line_table": True,
                }
            )

        for topic in sorted(set(topics)):
            if topic == -1:
                continue
            for term, weight in model.get_topic(topic)[:TOP_WORDS_PER_TOPIC]:
                term = normalize_term(term)
                if not is_useful_topic_word(term, seed_set):
                    continue
                rows.append(
                    {
                        "category": category,
                        "term": term,
                        "source": "bertopic_topic_word",
                        "topic": topic,
                        "weight": weight,
                        "used_in_line_table": term in CURATED_TOPIC_TERMS_FOR_LINE_TABLE[category],
                    }
                )

    terms = pd.DataFrame(rows).drop_duplicates(subset=["category", "term", "source"])
    return terms.sort_values(["category", "source", "topic", "term"], na_position="first")


def compile_category_patterns(terms: pd.DataFrame) -> dict[str, re.Pattern]:
    patterns = {}
    selected_terms = terms.loc[terms["used_in_line_table"] == True].copy()
    for category, group in selected_terms.groupby("category"):
        escaped_terms = sorted(
            {re.escape(term) for term in group["term"].dropna() if term},
            key=len,
            reverse=True,
        )
        patterns[category] = re.compile(r"\b(" + "|".join(escaped_terms) + r")\b", re.IGNORECASE)
    return patterns


def build_line_table(calls: pd.DataFrame, terms: pd.DataFrame) -> pd.DataFrame:
    patterns = compile_category_patterns(terms)
    rows = []

    for call_id, row in calls.iterrows():
        lines = str(row["transcript"]).splitlines()
        for line_number, line in enumerate(lines, start=1):
            line_text = line.strip()
            if not line_text:
                continue

            for category, pattern in patterns.items():
                matches = sorted({match.group(0).lower() for match in pattern.finditer(line_text)})
                if not matches:
                    continue

                rows.append(
                    {
                        "date": row["date"],
                        "exchange": row["exchange"],
                        "company": row["ticker"],
                        "category of keywords": category,
                        "line number": line_number,
                        "text/content of keywords": line_text,
                    }
                )

    return pd.DataFrame(
        rows,
        columns=[
            "date",
            "exchange",
            "company",
            "category of keywords",
            "line number",
            "text/content of keywords",
        ],
    )


def main() -> None:
    calls = pd.read_csv(CALLS_CSV)
    calls["date"] = clean_date_column(calls["date"])
    calls["exchange"] = normalize_exchange(calls["exchange"])

    segments = pd.read_csv(SEGMENTS_CSV)

    terms = build_topic_expanded_terms(segments)
    terms.to_csv(OUTPUT_TERMS, index=False)

    line_table = build_line_table(calls, terms)
    line_table.to_csv(OUTPUT_TABLE, index=False)

    print(f"Saved {len(terms):,} seed/topic-expanded terms to {OUTPUT_TERMS}")
    print(f"Saved {len(line_table):,} line-level rows to {OUTPUT_TABLE}")
    print()
    print("Rows by category:")
    print(line_table["category of keywords"].value_counts().to_string())


if __name__ == "__main__":
    main()
