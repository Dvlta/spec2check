"""Run every saved generated check for the ambiguous criteria on the 4,000-part synthetic population.

Reports, per criterion, disagreement between generated checks (median and max over pairs) next to the
disagreement between our reference readings, plus counts of clarification requests and execution failures.
Verdict vectors are cached in out/gen_verdicts.json.
"""
import concurrent.futures as cf
import itertools
import json
import random
import statistics
from pathlib import Path

from bench import make_part, specs
from harness import execute, parse

OUT = Path(__file__).parent / "out"
CODING = ("direct", "plain", "plain2", "plain3")
TRUNCATED = {"MAX_TOKENS", "max_tokens", "incomplete"}


def population():
    rng = random.Random(7)
    return [make_part(rng, f"p{i}", n_mean=10, size_scale=0.12, cluster_p=0.3, scale=rng.choice([1.0, 0.5])) for i in range(4000)]


def main():
    S = {s.id: s for s in specs() if s.tier == "T4"}
    latest = {}
    for r in map(json.loads, filter(str.strip, (OUT / "responses.jsonl").open())):
        if r.get("stop") != "error":
            latest[r["key"]] = r
    resps = [r for r in latest.values() if r["spec"] in S and r["cond"] in CODING]
    cache_path = OUT / "gen_verdicts.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    pop = population()
    todo = []
    for r in resps:
        code, _, _ = parse(r["text"])
        if code is not None and r["key"] not in cache:
            todo.append((r["key"], code))
    print(f"{len(todo)} programs to run", flush=True)

    def run(job):
        key, code = job
        got, err = execute(code, pop)
        return key, (got if got is not None else f"FAIL: {err}")

    with cf.ThreadPoolExecutor(8) as ex:
        for i, (key, v) in enumerate(ex.map(run, todo), 1):
            cache[key] = v
            if i % 50 == 0:
                print(f"{i}/{len(todo)}", flush=True)
                cache_path.write_text(json.dumps(cache))
    cache_path.write_text(json.dumps(cache))

    results = [json.loads(l) for l in (OUT / "results.jsonl").open() if l.strip()]
    reading = {r["key"]: r.get("reading") for r in results}
    rows = {}
    for sid, s in S.items():
        rs = [r for r in resps if r["spec"] == sid]
        no_code = [r for r in rs if parse(r["text"])[0] is None]
        truncated = sum(1 for r in no_code if r.get("stop") in TRUNCATED)  # cut off at the output-token limit
        asks = len(no_code) - truncated
        vecs, fails = {}, 0
        for r in rs:
            v = cache.get(r["key"])
            if v is None:
                continue
            if isinstance(v, str) or sum(x in ("ERR", "TYPE") for x in v) > 0.05 * len(v):
                fails += 1
                continue
            vecs[r["key"]] = v
        pair = [sum(a != b for a, b in zip(vecs[x], vecs[y])) / len(pop) for x, y in itertools.combinations(vecs, 2)]
        refv = {k: [f(p) for p in pop] for k, f in s.refs.items()}
        ref_max = max(sum(a != b for a, b in zip(refv[x], refv[y])) for x, y in itertools.combinations(refv, 2)) / len(pop)
        worst = max(itertools.combinations(vecs, 2), key=lambda xy: sum(a != b for a, b in zip(vecs[xy[0]], vecs[xy[1]])), default=None)
        rows[sid] = dict(n_code=len(vecs), asks=asks, truncated=truncated, fails=fails, ref_max=ref_max,
                         gen_median=statistics.median(pair) if pair else 0.0, gen_max=max(pair) if pair else 0.0,
                         worst_pair=[(k, reading.get(k)) for k in worst] if worst else None)
        print(f"{sid}: code {len(vecs)} ask {asks} truncated {truncated} fail {fails} | ref max {ref_max:.3f} | gen median {rows[sid]['gen_median']:.3f} "
              f"max {rows[sid]['gen_max']:.3f} | worst {rows[sid]['worst_pair']}", flush=True)
    (OUT / "gen_disagreement.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
