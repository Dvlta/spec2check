"""Agreement between the author's blind labels (human_labels.json exported from the sheet) and the LLM judges.

Usage: python human_agreement.py [path/to/human_labels.json]   (default out/human_labels.json)
"""
import json
import sys
from pathlib import Path

from judge import JUDGES, majority

OUT = Path(__file__).parent / "out"


def kappa(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return po, (po - pe) / (1 - pe) if pe < 1 else float("nan")


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT / "human_labels.json"
    H = json.loads(path.read_text())
    H = H.get("labels", H)  # accept a wrapper with annotator metadata
    key = json.loads((OUT / "human_sample_key.json").read_text())
    J = majority()
    rows = [(key[i], h) for i, h in H.items() if "states" in h and "flags" in h]
    print(f"{len(rows)} labeled items")
    for idx, q in ((0, "states"), (1, "flags")):
        h = [lab[q] for _, lab in rows]
        print(f"{q}: human YES rate {sum(h) / len(h):.2f}")
        for judge in JUDGES:
            j = [J[k][judge][idx] for k, _ in rows]
            po, k = kappa(h, j)
            print(f"  vs {judge:18s} agree {sum(x == y for x, y in zip(h, j))}/{len(h)} ({po:.2f})  kappa {k:.2f}")
        both = [(x, J[k][JUDGES[0]][idx]) for (k, _), x in zip(rows, h) if J[k][JUDGES[0]][idx] == J[k][JUDGES[1]][idx]]
        print(f"  vs judges where they agree: {sum(x == y for x, y in both)}/{len(both)}")
    print("\nDisagreements (item, model, human states/flags, judges):")
    for i, h in H.items():
        if "states" not in h or "flags" not in h:
            continue
        k = key[i]
        js = {judge: J[k][judge][:2] for judge in JUDGES}
        if any((h["states"], h["flags"]) != v for v in js.values()):
            print(f"  #{i} {k}  human {h['states']}/{h['flags']}  " + "  ".join(f"{j.split('-')[0]} {v[0]}/{v[1]}" for j, v in js.items())
                  + (f"  note: {h['note']}" if h.get("note") else ""))


if __name__ == "__main__":
    main()
