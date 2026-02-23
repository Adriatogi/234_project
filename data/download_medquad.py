"""
Download MedQuAD dataset from GitHub and convert XML to a flat CSV.

Outputs:
    medquad.csv — all question-answer pairs across all 12 NIH sources

Each row contains:
    - source: the NIH source directory (e.g. '5_NIDDK_QA')
    - document_id: XML document ID
    - focus: the disease/condition name
    - url: source URL from NIH
    - question_id: unique question ID (e.g. '0000001-1')
    - question_type: category (information, symptoms, causes, treatment, etc.)
    - question: the question text
    - answer: the answer text

Source: https://github.com/abachaa/MedQuAD
"""

import os
import subprocess
from xml.etree import ElementTree as ET
import pandas as pd
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
CLONE_DIR = os.path.join(OUTPUT_DIR, "_medquad_repo")
REPO_URL = "https://github.com/abachaa/MedQuAD.git"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "medquad.csv")


def parse_xml_file(filepath, source_name):
    """Parse a single MedQuAD XML file into a list of QA pair dicts."""
    rows = []
    tree = ET.parse(filepath)
    root = tree.getroot()

    doc_id = root.attrib.get("id", "")
    doc_url = root.attrib.get("url", "")
    focus_el = root.find("Focus")
    focus = focus_el.text.strip() if focus_el is not None and focus_el.text else ""

    qa_pairs = root.find("QAPairs")
    if qa_pairs is None:
        return rows

    for qapair in qa_pairs.findall("QAPair"):
        question_el = qapair.find("Question")
        answer_el = qapair.find("Answer")

        if question_el is None or answer_el is None:
            continue

        rows.append({
            "source": source_name,
            "document_id": doc_id,
            "focus": focus,
            "url": doc_url,
            "question_id": question_el.attrib.get("qid", ""),
            "question_type": question_el.attrib.get("qtype", ""),
            "question": question_el.text.strip() if question_el.text else "",
            "answer": answer_el.text.strip() if answer_el.text else "",
        })

    return rows


def main():
    # Clone the repo if not already present
    if not os.path.exists(CLONE_DIR):
        print("Cloning MedQuAD repo...")
        subprocess.run(
            ["git", "clone", "--depth", "1", REPO_URL, CLONE_DIR],
            check=True,
        )
    else:
        print(f"MedQuAD repo already cloned at {CLONE_DIR}")

    all_rows = []

    for source_name in sorted(os.listdir(CLONE_DIR)):
        source_path = os.path.join(CLONE_DIR, source_name)
        if not os.path.isdir(source_path) or source_name.startswith("."):
            continue

        xml_files = [f for f in os.listdir(source_path) if f.endswith(".xml")]
        if not xml_files:
            continue

        for xml_file in sorted(xml_files):
            filepath = os.path.join(source_path, xml_file)
            rows = parse_xml_file(filepath, source_name)
            all_rows.extend(rows)

        print(f"  {source_name}: {len(xml_files)} files")

    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\n  Total QA pairs: {len(df)}")
    print(f"  Sources: {df['source'].nunique()}")
    print(f"  Saved to {OUTPUT_PATH}")

    # Print breakdown by source
    print(f"\n--- Breakdown by source ---")
    for source, count in df.groupby("source").size().items():
        print(f"  {source}: {count} QA pairs")

    # Print sample
    print(f"\n--- Sample row ---")
    row = df.iloc[0]
    print(f"  Source:   {row['source']}")
    print(f"  Focus:    {row['focus']}")
    print(f"  QType:    {row['question_type']}")
    print(f"  Question: {row['question'][:150]}")
    print(f"  Answer:   {row['answer'][:150]}...")


if __name__ == "__main__":
    main()
