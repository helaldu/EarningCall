import re
from pathlib import Path

import pandas as pd
from pypdf import PdfReader


PDF_DIR = Path("Landmark Studies & Preliminary Key Words")
OUTPUT_CSV = Path("landmark_pdf_keyword_table.csv")

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


def compile_patterns() -> dict[str, re.Pattern]:
    patterns = {}
    for category, keywords in KEYWORD_GROUPS.items():
        escaped = sorted((re.escape(word) for word in keywords), key=len, reverse=True)
        patterns[category] = re.compile(r"\b(" + "|".join(escaped) + r")\b", re.IGNORECASE)
    return patterns


def clean_line(line: str) -> str:
    line = line.replace("\u00ad", "")
    line = re.sub(r"\s+", " ", line)
    return line.strip()


def document_id_from_path(path: Path) -> str:
    match = re.match(r"^(\d+)\.", path.name)
    if match:
        return match.group(1)
    return path.stem


def extract_rows_from_pdf(path: Path, patterns: dict[str, re.Pattern]) -> list[dict]:
    reader = PdfReader(path)
    rows = []
    document_id = document_id_from_path(path)

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        for line_number, raw_line in enumerate(text.splitlines(), start=1):
            line = clean_line(raw_line)
            if not line:
                continue

            for category, pattern in patterns.items():
                words_found = sorted({match.group(0).lower() for match in pattern.finditer(line)})
                for word in words_found:
                    rows.append(
                        {
                            "document id": document_id,
                            "keyword category": category,
                            "the word found": word,
                            "line/page number": f"page {page_number}, line {line_number}",
                            "text line": line,
                        }
                    )

    return rows


def main() -> None:
    patterns = compile_patterns()
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    rows = []

    for pdf in pdfs:
        rows.extend(extract_rows_from_pdf(pdf, patterns))

    table = pd.DataFrame(
        rows,
        columns=[
            "document id",
            "keyword category",
            "the word found",
            "line/page number",
            "text line",
        ],
    )
    table.to_csv(OUTPUT_CSV, index=False)

    print(f"Read {len(pdfs)} PDF files from {PDF_DIR}")
    print(f"Saved {len(table):,} rows to {OUTPUT_CSV}")
    if not table.empty:
        print()
        print("Rows by category:")
        print(table["keyword category"].value_counts().to_string())
        print()
        print("Rows by document:")
        print(table["document id"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
