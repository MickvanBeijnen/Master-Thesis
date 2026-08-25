"""
msorcas_generator.py

Script that generates a new dataset (called MS-ORCAS), which is the subset of MS-MARCO documents 
who are referenced by the ORCAS dataset.

Running this script is useful, as the resulting MS-ORCAS dataset is drastically smaller in size 
compared to the MS-MARCO dataset, resulting in less storage space, as well as computations
being needed.

"""

from pathlib import Path

ORCAS_PATH = Path("../Datasets/orcas.tsv")
MSMARCO_PATH = Path("../Datasets/msmarco-docs.tsv")
OUTPUT_PATH = Path("../Datasets/msorcas.tsv")

def load_orcas_docids(orcas_path: Path) -> set[str]:
    """Load all unique DIDs from ORCAS into a set."""
    docids = set()

    with orcas_path.open("r", encoding="utf-8", errors="replace") as f:
        for line_number, line in enumerate(f, start=1):
            parts = line.rstrip("\n").split("\t", 3)
            if len(parts) != 4:
                continue

            docid = parts[2]
            docids.add(docid)

            if line_number % 1_000_000 == 0:
                print(f"Read {line_number:,} ORCAS rows... unique DIDs so far: {len(docids):,}")

    return docids

def filter_msmarco(msmarco_path: Path, output_path: Path, target_docids: set[str]) -> None:
    """Write only MSMARCO rows whose DID appears in ORCAS."""
    written = 0
    scanned = 0

    with (
        msmarco_path.open("r", encoding="utf-8", errors="replace") as infile,
        output_path.open("w", encoding="utf-8", errors="replace", newline="") as outfile,
    ):
        for line in infile:
            scanned += 1

            parts = line.split("\t", 1)
            if len(parts) < 2:
                continue

            docid = parts[0]
            if docid in target_docids:
                outfile.write(line)
                written += 1

            if scanned % 500_000 == 0:
                print(
                    f"Scanned {scanned:,} MSMARCO rows... "
                    f"written {written:,} matching rows"
                )

    print(f"\nFinished.")
    print(f"Total MSMARCO rows scanned : {scanned:,}")
    print(f"Total subset rows written  : {written:,}")
    print(f"Output written to          : {output_path}")

def main():
    print("Loading ORCAS DIDs...")
    orcas_docids = load_orcas_docids(ORCAS_PATH)
    print(f"Loaded {len(orcas_docids):,} unique ORCAS DIDs.\n")

    print("Filtering MSMARCO...")
    filter_msmarco(MSMARCO_PATH, OUTPUT_PATH, orcas_docids)

if __name__ == "__main__":
    main()
