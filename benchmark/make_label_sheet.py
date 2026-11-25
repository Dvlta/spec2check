"""Build a blind labeling sheet for checking the LLM judges against a human annotator.

Samples 60 judged responses (20 per model, fixed seed), shuffles them, hides the model and the judges' labels,
and writes out/human_label_sheet.html. The item -> response key mapping goes to out/human_sample_key.json,
which the sheet never reads. Labels are exported from the sheet as human_labels.json; score with human_agreement.py.
"""
import json
import random
from pathlib import Path

from bench import specs
from judge import JUDGES, PROMPT, majority

OUT = Path(__file__).parent / "out"
PER_MODEL = 20
SEED = 2027

GUIDE = PROMPT.split("Answer two questions", 1)[1].split("Reply with exactly", 1)[0].strip()
GUIDE = "Answer two questions" + GUIDE


def main():
    S = {s.id: s for s in specs()}
    J = majority()
    latest = {}
    for r in map(json.loads, filter(str.strip, (OUT / "responses.jsonl").open())):
        if r.get("stop") != "error":
            latest[r["key"]] = r
    judged = [r for r in latest.values() if r["key"] in J and all(j in J[r["key"]] for j in JUDGES)]
    rng = random.Random(SEED)
    sample = []
    for m in sorted({r["model"] for r in judged}):
        pool = sorted((r for r in judged if r["model"] == m), key=lambda r: r["key"])
        sample += rng.sample(pool, PER_MODEL)
    rng.shuffle(sample)
    items = [dict(id=i + 1, spec=S[r["spec"]].text, ambiguity=S[r["spec"]].ambiguity, response=r["text"][:12000])
             for i, r in enumerate(sample)]
    (OUT / "human_sample_key.json").write_text(json.dumps({i + 1: r["key"] for i, r in enumerate(sample)}, indent=1))
    data = json.dumps(dict(guide=GUIDE, items=items)).replace("</", "<\\/")
    html = (Path(__file__).parent / "label_sheet_template.html").read_text().replace("__DATA__", data)
    (OUT / "human_label_sheet.html").write_text(html)
    print(f"{len(items)} items -> {OUT / 'human_label_sheet.html'}")


if __name__ == "__main__":
    main()
