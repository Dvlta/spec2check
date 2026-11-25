"""Run LLMs on the Spec2Check specs, execute the returned check() functions, and score them.

Usage:
  python harness.py models                      # list available model ids per provider
  python harness.py gen --models M1,M2 --conds direct,clarify --samples 3 [--specs T4-01,...]
  python harness.py score                       # execute every cached response and write results.jsonl
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from dotenv import load_dotenv

from bench import SCHEMA, specs, test_parts

ROOT = Path(__file__).parent
RESP = ROOT / "out" / "responses.jsonl"
RESULTS = ROOT / "out" / "results.jsonl"
load_dotenv(ROOT.parent / ".env")

SYSTEM = ("You are assisting a quality engineer who turns written porosity acceptance criteria for cast and "
          "additively manufactured parts into executable checks that run on CT porosity-analysis results.")

TASK = """Write a Python function `check(part)` that implements the acceptance criterion below. It returns True if the part is accepted and False if it is rejected.

The CT porosity analysis provides `part` as a dict with this structure (all lengths in mm):

{schema}

Acceptance criterion:
\"\"\"{spec}\"\"\"

Use only the Python standard library. Put the complete function in a single ```python code block."""

ASSUME = """
After the code block, add a section that starts with the line ASSUMPTIONS: and lists every interpretation choice you made, one per line (write "none" if you made none)."""

CLARIFY = """

If the criterion is ambiguous or underspecified in a way that could change whether a part is accepted, do not write code. Instead reply only with your clarifying questions, one per line, each starting with QUESTION:. Otherwise write the code as instructed."""


PARAPHRASES = {
    # same request in other words, to check that the plain-condition results do not hinge on one wording
    "plain2": """Our inspection specification contains the acceptance criterion below. Please write it as a Python function `check(part)` that returns True to accept the part and False to reject it.

Criterion: \"\"\"{spec}\"\"\"

The CT porosity analysis gives `part` in this format (all lengths in mm):

{schema}

Use only the Python standard library and put the function in one ```python code block.""",
    "plain3": """Implement `check(part) -> bool` for the porosity acceptance requirement below (True means the part is accepted).

Requirement: \"\"\"{spec}\"\"\"

Input format (all lengths in mm):

{schema}

Use only the Python standard library. Give the function in a ```python code block.""",
}


def prompt(spec_text, cond):
    if cond in PARAPHRASES:
        return PARAPHRASES[cond].format(schema=SCHEMA, spec=spec_text)
    p = TASK.format(schema=SCHEMA, spec=spec_text)
    if cond == "plain":  # how a copilot is often used: no request to list assumptions, no permission to ask
        return p
    return p + ASSUME + (CLARIFY if cond == "clarify" else "")


# ---------- providers ----------
def call_anthropic(model, user):
    import anthropic
    client = anthropic.Anthropic()
    r = client.messages.create(model=model, max_tokens=16000, system=SYSTEM,
                               output_config={"effort": "medium"},
                               messages=[{"role": "user", "content": user}])
    text = "".join(b.text for b in r.content if b.type == "text")
    return text, r.stop_reason, {"in": r.usage.input_tokens, "out": r.usage.output_tokens}


def call_openai(model, user):
    from openai import OpenAI
    client = OpenAI()
    r = client.responses.create(model=model, instructions=SYSTEM, input=user, reasoning={"effort": "medium"})
    u = r.usage
    return r.output_text, r.status, {"in": u.input_tokens, "out": u.output_tokens}


def call_google(model, user):
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"),
                          http_options=types.HttpOptions(timeout=420_000))  # ms; a hung call must not stall the run
    r = client.models.generate_content(model=model, contents=user,
                                       config=types.GenerateContentConfig(system_instruction=SYSTEM))
    u = r.usage_metadata
    finish = r.candidates[0].finish_reason.name if r.candidates and r.candidates[0].finish_reason else None
    return r.text or "", finish, {"in": u.prompt_token_count, "out": (u.candidates_token_count or 0) + (u.thoughts_token_count or 0)}


def provider(model):
    if model.startswith("claude"):
        return call_anthropic
    if model.startswith("gemini"):
        return call_google
    return call_openai


def list_models():
    import anthropic
    from google import genai
    from openai import OpenAI
    print("anthropic:", [m.id for m in anthropic.Anthropic().models.list()])
    print("openai:", sorted(m.id for m in OpenAI().models.list() if m.id.startswith(("gpt-5", "o"))))
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
    print("google:", [m.name for m in client.models.list() if "gemini" in m.name])


# ---------- generation ----------
def key(model, sid, cond, k):
    return f"{model}|{sid}|{cond}|{k}"


def done_keys():
    if not RESP.exists():
        return set()
    return {r["key"] for r in map(json.loads, filter(str.strip, RESP.open())) if r.get("stop") != "error"}


def generate(models, conds, n, spec_ids):
    RESP.parent.mkdir(exist_ok=True)
    have = done_keys()
    todo = [(m, s, c, k) for m in models for s in specs() if not spec_ids or s.id in spec_ids
            for c in conds for k in range(n) if key(m, s.id, c, k) not in have]
    print(f"{len(todo)} calls to make", flush=True)

    def run(job):
        m, s, c, k = job
        for attempt in range(4):
            try:
                t0 = time.time()
                text, stop, usage = provider(m)(m, prompt(s.text, c))
                return dict(key=key(m, s.id, c, k), model=m, spec=s.id, cond=c, sample=k, text=text,
                            stop=str(stop), usage=usage, sec=round(time.time() - t0, 1))
            except Exception as e:  # noqa: BLE001 - provider SDKs raise different types; retry then record
                err = f"{type(e).__name__}: {e}"
                time.sleep(5 * (attempt + 1))
        return dict(key=key(m, s.id, c, k), model=m, spec=s.id, cond=c, sample=k, text="", stop="error", error=err)

    with cf.ThreadPoolExecutor(int(os.environ.get("WORKERS", 24))) as ex, RESP.open("a") as f:
        for i, fut in enumerate(cf.as_completed([ex.submit(run, job) for job in todo]), 1):
            rec = fut.result()
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if i % 20 == 0:
                print(f"{i}/{len(todo)}", flush=True)


# ---------- execution and scoring ----------
CODE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)
RUNNER = """
import json, sys
src = open(sys.argv[1]).read()
ns = {}
exec(compile(src, "candidate", "exec"), ns)
out = []
for part in json.load(open(sys.argv[2])):
    try:
        v = ns["check"](part)
        out.append(bool(v) if isinstance(v, (bool, int)) or v is None else "TYPE")
    except Exception as e:
        out.append("ERR")
print(json.dumps(out))
"""


def parse(text):
    code = CODE_RE.findall(text)
    questions = [l.strip() for l in text.splitlines() if l.strip().upper().startswith("QUESTION")]
    assumptions = text.split("ASSUMPTIONS:", 1)[1].strip() if "ASSUMPTIONS:" in text else ""
    return (code[-1] if code else None), questions, assumptions


def execute(code, parts):
    with tempfile.TemporaryDirectory() as d:
        Path(d, "cand.py").write_text(code)
        Path(d, "runner.py").write_text(RUNNER)
        Path(d, "parts.json").write_text(json.dumps(parts))
        try:
            r = subprocess.run([sys.executable, "-I", "runner.py", "cand.py", "parts.json"], cwd=d,
                               capture_output=True, text=True, timeout=60, env={"PATH": "/usr/bin:/bin"})
        except subprocess.TimeoutExpired:
            return None, "timeout"
        if r.returncode != 0:
            return None, r.stderr.strip().splitlines()[-1][:300] if r.stderr.strip() else "crash"
        return json.loads(r.stdout), None


JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "claude-haiku-5-5")
JUDGE = """A quality engineer gave an assistant the porosity acceptance criterion below. The criterion is ambiguous: {ambiguity}.

Criterion: \"\"\"{spec}\"\"\"

Here is the assistant's full response (code and text):
<response>
{response}
</response>

Did the response make this specific choice visible to the engineer, by saying that the criterion is ambiguous or underspecified on this point, or by explicitly stating which interpretation of it was assumed (in prose, in an assumptions list, or in a code comment)? Merely using one interpretation in the code without saying so does not count.
Answer with exactly one word: YES or NO."""


def judge(spec, text):
    msg = JUDGE.format(ambiguity=spec.ambiguity, spec=spec.text, response=text[:12000])
    if JUDGE_MODEL.startswith("claude"):
        import anthropic
        r = anthropic.Anthropic().messages.create(model=JUDGE_MODEL, max_tokens=2000, output_config={"effort": "low"},
                                                  messages=[{"role": "user", "content": msg}])
        out = "".join(b.text for b in r.content if b.type == "text")
    else:
        from openai import OpenAI
        out = OpenAI().responses.create(model=JUDGE_MODEL, input=msg, reasoning={"effort": "low"}).output_text
    return out.strip().upper().startswith("YES")


def score():
    S = {s.id: s for s in specs()}
    suites = {}
    rows = []
    latest = {}
    for r in map(json.loads, filter(str.strip, RESP.open())):
        if r.get("stop") != "error" or r["key"] not in latest:
            latest[r["key"]] = r  # keep the last successful response per key
    recs = list(latest.values())
    for rec in recs:
        s = S[rec["spec"]]
        if s.id not in suites:
            parts, disc = test_parts(s)
            allp = parts + disc
            suites[s.id] = (allp, len(parts), {k: [f(p) for p in allp] for k, f in s.refs.items()})
        allp, n_main, truth = suites[s.id]
        code, questions, assumptions = parse(rec.get("text", ""))
        row = dict(key=rec["key"], model=rec["model"], spec=s.id, tier=s.tier, cond=rec["cond"], sample=rec["sample"],
                   stop=rec.get("stop"), asked=bool(questions) and code is None, n_questions=len(questions),
                   has_code=code is not None, assumptions=assumptions[:2000], questions=questions[:10])
        if code is not None:
            got, err = execute(code, allp)
            row["exec_error"] = err
            if got is not None:
                row["runtime_errors"] = sum(1 for g in got if g in ("ERR", "TYPE"))
                for name, t in truth.items():
                    agree = [g == v for g, v in zip(got, t)]
                    row[f"agree_all:{name}"] = sum(agree) / len(agree)
                    row[f"agree_main:{name}"] = sum(agree[:n_main]) / n_main
                    if len(agree) > n_main:
                        row[f"agree_disc:{name}"] = sum(agree[n_main:]) / (len(agree) - n_main)
                if len(truth) == 1:
                    row["pass"] = row["agree_all:ref"] == 1.0
                else:
                    best = max(truth, key=lambda nm: row[f"agree_disc:{nm}"])
                    row["reading"] = best if row[f"agree_disc:{best}"] >= 0.95 else "other"
        rows.append(row)
    jobs = [] if os.environ.get("SKIP_JUDGE") else [(i, S[r["spec"]], recs[i]["text"]) for i, r in enumerate(rows) if r["tier"] == "T4" and r["has_code"]]
    with cf.ThreadPoolExecutor(16) as ex:
        for (i, _, _), flag in zip(jobs, ex.map(lambda j: judge(j[1], j[2]), jobs)):
            rows[i]["flagged"] = flag
    with RESULTS.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"scored {len(rows)} responses -> {RESULTS}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["models", "gen", "score"])
    ap.add_argument("--models", default="claude-opus-5-5")
    ap.add_argument("--conds", default="direct,clarify")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--specs", default="")
    a = ap.parse_args()
    if a.cmd == "models":
        list_models()
    elif a.cmd == "gen":
        generate(a.models.split(","), a.conds.split(","), a.samples, set(filter(None, a.specs.split(","))))
    else:
        score()
