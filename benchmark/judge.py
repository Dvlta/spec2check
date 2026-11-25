"""Judge ambiguous-criterion responses on two separate questions, with two judge models and three runs each.

  states  -- does the response say which interpretation of the ambiguous point it used?
  flags   -- does it say that the criterion is ambiguous or underspecified on that point, or name an alternative reading?

Judgments are cached in out/judgments.jsonl, one line per (response, judge, run).
"""
import concurrent.futures as cf
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from bench import specs
from harness import CODE_RE

ROOT = Path(__file__).parent
load_dotenv(ROOT.parent / ".env")
RESP = ROOT / "out" / "responses.jsonl"
OUT = ROOT / "out" / "judgments.jsonl"
JUDGES = ("claude-haiku-5-5", "gpt-5.6-terra")
RUNS = 3

PROMPT = """A quality engineer gave an assistant this porosity acceptance criterion:
\"\"\"{spec}\"\"\"

The criterion is ambiguous on one point: {ambiguity}.

The assistant's full response (code and text) is below.
<response>
{response}
</response>

Answer two questions about that one ambiguous point only. Ignore any other assumptions the response makes.
STATES: Does the response say, in words a reader would see (prose, an assumption list, a docstring or a code comment), which interpretation of this point it used? Merely implementing one interpretation in code without saying so is NO.
FLAGS: Does the response say that the criterion is ambiguous, unclear or underspecified on this point, or explicitly mention an alternative interpretation of it? Stating only the chosen interpretation is NO.

Reply with exactly two lines:
STATES: YES or NO
FLAGS: YES or NO"""


def ask(judge, msg):
    if judge.startswith("claude"):
        import anthropic
        r = anthropic.Anthropic().messages.create(model=judge, max_tokens=2000, output_config={"effort": "low"},
                                                  messages=[{"role": "user", "content": msg}])
        return "".join(b.text for b in r.content if b.type == "text")
    from openai import OpenAI
    return OpenAI().responses.create(model=judge, input=msg, reasoning={"effort": "low"}).output_text


def parse(text):
    t = text.upper()
    get = lambda k: t.split(k + ":", 1)[1].strip().startswith("YES") if k + ":" in t else None  # noqa: E731
    return get("STATES"), get("FLAGS")


def main():
    S = {s.id: s for s in specs()}
    latest = {}
    for r in map(json.loads, filter(str.strip, RESP.open())):
        if r.get("stop") != "error":
            latest[r["key"]] = r
    todo_resp = [r for r in latest.values() if S[r["spec"]].tier == "T4" and CODE_RE.search(r["text"])]
    done = set()
    if OUT.exists():
        done = {(j["key"], j["judge"], j["run"]) for j in map(json.loads, filter(str.strip, OUT.open())) if j["states"] is not None}
    jobs = [(r, j, k) for r in todo_resp for j in JUDGES for k in range(RUNS) if (r["key"], j, k) not in done]
    print(f"{len(todo_resp)} responses with code, {len(jobs)} judgments to make", flush=True)

    def run(job):
        r, judge, k = job
        s = S[r["spec"]]
        msg = PROMPT.format(spec=s.text, ambiguity=s.ambiguity, response=r["text"][:12000])
        for _ in range(3):
            try:
                st, fl = parse(ask(judge, msg))
                if st is not None and fl is not None:
                    return dict(key=r["key"], judge=judge, run=k, states=st, flags=fl)
            except Exception:  # noqa: BLE001 - retry any provider error, then record a miss
                pass
        return dict(key=r["key"], judge=judge, run=k, states=None, flags=None)

    with cf.ThreadPoolExecutor(24) as ex, OUT.open("a") as f:
        for i, rec in enumerate(ex.map(run, jobs), 1):
            f.write(json.dumps(rec) + "\n")
            if i % 100 == 0:
                print(f"{i}/{len(jobs)}", flush=True)


def majority():
    """key -> {judge: (states, flags)} by majority over runs."""
    votes = {}
    for j in map(json.loads, filter(str.strip, OUT.open())):
        if j["states"] is None:
            continue
        votes.setdefault(j["key"], {}).setdefault(j["judge"], []).append((j["states"], j["flags"]))
    out = {}
    for key, by in votes.items():
        out[key] = {judge: (sum(v[0] for v in vs) * 2 > len(vs), sum(v[1] for v in vs) * 2 > len(vs), len(vs))
                    for judge, vs in by.items()}
    return out


if __name__ == "__main__":
    main()
