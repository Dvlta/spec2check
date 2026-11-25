"""Final analysis for the paper: correctness, asking, silent and unflagged choices with spec-level intervals, observed flips."""
import collections
import json
import random
from pathlib import Path

from bench import make_part, specs
from judge import JUDGES, majority

OUT = Path(__file__).parent / "out"
MODELS = ("claude-opus-5-5", "gpt-5.6-sol", "gemini-4-neon")
S = {s.id: s for s in specs()}
T4 = [s.id for s in specs() if s.tier == "T4"]
TRUNCATED = {"MAX_TOKENS", "max_tokens", "incomplete"}  # cut off at the output-token limit, not a clarification


def rows():
    return [json.loads(l) for l in (OUT / "results.jsonl").open() if l.strip()]


def boot_ci(per_spec, n=5000, seed=0):
    """95% interval for a pooled rate, resampling criteria (the samples within a criterion are correlated)."""
    rng = random.Random(seed)
    ids = [k for k, v in per_spec.items() if v]
    stats = []
    for _ in range(n):
        pick = [rng.choice(ids) for _ in ids]
        num = sum(sum(per_spec[i]) for i in pick)
        den = sum(len(per_spec[i]) for i in pick)
        stats.append(num / den)
    stats.sort()
    return stats[int(0.025 * n)], stats[int(0.975 * n)]


def main():
    R = rows()
    J = majority()
    print("== Clear criteria (assumptions + plain): correct, wrong verdicts, errors ==")
    for m in MODELS:
        rs = [r for r in R if r["model"] == m and r["tier"] != "T4" and r["cond"] in ("direct", "plain") and r["has_code"]]
        wrong = sum(1 for r in rs if not r.get("pass") and not r.get("runtime_errors") and not r.get("exec_error"))
        err = sum(1 for r in rs if not r.get("pass") and (r.get("runtime_errors") or r.get("exec_error")))
        print(f"  {m:18s} correct {sum(r.get('pass', False) for r in rs)}/{len(rs)}  wrong-verdict {wrong}  error {err}")
    print("\n== Clarify condition: asked ==")
    for m in MODELS:
        a = [r for r in R if r["model"] == m and r["tier"] == "T4" and r["cond"] == "clarify"]
        c = [r for r in R if r["model"] == m and r["tier"] != "T4" and r["cond"] == "clarify"]
        print(f"  {m:18s} ambiguous {sum(r['asked'] for r in a)}/{len(a)}   clear {sum(r['asked'] for r in c)}/{len(c)}")
    for what, idx in (("SILENT (code, interpretation not stated)", 0), ("UNFLAGGED (code, ambiguity not mentioned)", 1)):
        print(f"\n== {what}; per judge (majority of 3 runs); 95% CI over criteria ==")
        for cond in ("direct", "plain", "plain2", "plain3"):
            for m in MODELS:
                cells = []
                for judge in JUDGES:
                    per = collections.defaultdict(list)
                    for r in R:
                        if r["model"] == m and r["tier"] == "T4" and r["cond"] == cond:
                            if not r["has_code"]:
                                per[r["spec"]].append(False)
                            elif r["key"] in J and judge in J[r["key"]]:
                                per[r["spec"]].append(not J[r["key"]][judge][idx])
                    k = sum(sum(v) for v in per.values()); n = sum(len(v) for v in per.values())
                    lo, hi = boot_ci(per) if n else (0, 0)
                    specs_hit = sum(any(v) for v in per.values())
                    cells.append(f"{judge.split('-')[0]}: {k}/{n} [{lo:.2f},{hi:.2f}] crit {specs_hit}/{len(per)}")
                print(f"  {cond:7s} {m:18s} " + " | ".join(cells))
    print("\n== FLAGGED OR ASKED (Table 2): coded and flagged, or a clarification request; truncated generations excluded ==")
    for cond in ("direct", "plain", "plain2", "plain3"):
        for m in MODELS:
            rs = [r for r in R if r["model"] == m and r["tier"] == "T4" and r["cond"] == cond]
            cells = []
            for judge in JUDGES:
                asked = sum(1 for r in rs if not r["has_code"] and r.get("stop") not in TRUNCATED)
                flagged = sum(1 for r in rs if r["has_code"] and r["key"] in J and judge in J[r["key"]] and J[r["key"]][judge][1])
                cells.append(f"{judge.split('-')[0]}: {asked + flagged}/{len(rs)}")
            print(f"  {cond:7s} {m:18s} " + " | ".join(cells))
    print("\n== Judge agreement (majority labels) ==")
    for idx, name in ((0, "states"), (1, "flags")):
        both = [v for v in J.values() if all(j in v for j in JUDGES)]
        agree = sum(v[JUDGES[0]][idx] == v[JUDGES[1]][idx] for v in both)
        print(f"  {name}: {agree}/{len(both)}")
    print("\n== Readings implemented, by model and condition ==")
    for sid in T4:
        line = []
        for m in MODELS:
            for cond in ("direct", "plain"):
                c = collections.Counter(r.get("reading", "no code") for r in R if r["model"] == m and r["spec"] == sid and r["cond"] == cond)
                line.append(f"{m.split('-')[0][:3]}/{cond[0]} {dict(c)}")
        print(f"  {sid}: " + "  ".join(line))
    print("\n== Verdict flips on 4,000 production-like parts ==")
    rng = random.Random(7)
    pop = [make_part(rng, f"p{i}", n_mean=10, size_scale=0.12, cluster_p=0.3, scale=rng.choice([1.0, 0.5])) for i in range(4000)]
    for sid in T4:
        s = S[sid]
        v = {k: [f(p) for p in pop] for k, f in s.refs.items()}
        names = list(v)
        worst = max(sum(a != b for a, b in zip(v[x], v[y])) for i, x in enumerate(names) for y in names[i + 1:]) / len(pop)
        seen = sorted({r.get("reading") for r in R if r["spec"] == sid and r.get("reading") in s.refs})
        obs = (sum(a != b for a, b in zip(v[seen[0]], v[seen[-1]])) / len(pop)) if len(seen) > 1 else 0.0
        print(f"  {sid}: any two readings {worst:.3f}; readings models chose {seen} -> {obs:.3f}")


if __name__ == "__main__":
    main()
