# Spec2Check

Code and data for *Prompting Makes Inspection Copilots State Their Assumptions, but Not Which Ones Matter*
(Leo Sun, AI4Factory workshop at AAAI 2027).

Spec2Check is a benchmark of 45 porosity acceptance criteria for industrial CT inspection. 24 are fully specified;
21 are ambiguous, and each of those has two or three plausible readings implemented as executable references.
LLM-generated `check(part)` functions are scored by executing them on synthetic CT pore tables.

## Layout

```
benchmark/
  bench.py              criteria, reference readings, part generator, test parts
  harness.py            prompts, model calls, execution sandbox, scoring
  judge.py              two-question LLM judge (states / flags), two judges x 3 runs, majority vote
  analyze2.py           all headline numbers in the paper (Table 2, results paragraphs)
  gen_disagreement.py   runs every generated check on the 4,000-part population (Table 1)
  sensitivity.py        reference-reading disagreement at median pore sizes 0.10 / 0.12 / 0.15 mm
  make_label_sheet.py   builds the blind human labeling sheet (60 responses, 20 per model)
  human_agreement.py    agreement between the human labels and the judges
  label_sheet_template.html
  out/
    responses.jsonl             all model responses (1,593 successful, plus a few retried errors)
    responses_v1_archive.jsonl  pilot responses before nine criteria were clarified
    results.jsonl               execution scores and matched readings per response
    judgments.jsonl             every judge call (response, judge, run, states, flags)
    gen_verdicts.json           verdict vectors of generated checks on the 4,000-part population
    gen_disagreement.json       per-criterion disagreement between generated checks (Table 1)
    human_label_sheet.html      the blind sheet the author labeled
    human_sample_key.json       item -> response key for that sheet
    human_labels.json           the author's labels
    analysis.txt, sensitivity.txt, human_agreement.txt   outputs of the scripts above
paper/
  main.tex, refs.bib, main.pdf  (compile with the AAAI 2027 author kit: aaai2027.sty / aaai2027.bst)
```

## Conditions

Condition names in the data differ from the paper:

| data      | paper |
|-----------|-------|
| `plain`, `plain2`, `plain3` | the three plain-request wordings |
| `direct`  | assumptions (asks for an `ASSUMPTIONS:` list) |
| `clarify` | clarify (allows clarifying questions instead of code) |

Models: `claude-opus-5-5` (effort medium), `gpt-5.6-sol` (reasoning medium), `gemini-4-neon` (default thinking).
Judges: `claude-haiku-5-5` and `gpt-5.6-terra`.

## Reproducing the paper's numbers from the released data (no API calls)

Python 3.12 with `pip install -r requirements.txt` (the analysis scripts import `python-dotenv` through `judge.py`; the provider SDKs are only used when calling models).

```
cd benchmark
python analyze2.py           # -> out/analysis.txt
python gen_disagreement.py   # uses the cached verdicts in out/gen_verdicts.json
python sensitivity.py
python human_agreement.py out/human_labels.json
```

## Re-running the models

Put API keys in a `.env` file at the repository root (it is git-ignored):

```
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
GEMINI_API_KEY=...
```

```
cd benchmark
python harness.py gen --models claude-opus-5-5,gpt-5.6-sol,gemini-4-neon --conds direct,clarify,plain,plain2,plain3 --samples 3
SKIP_JUDGE=1 python harness.py score
python judge.py
python analyze2.py
```

`harness.py gen` skips keys already present in `out/responses.jsonl`; move that file aside to start fresh.
Generated code is executed in a separate Python process with only the standard library and no credentials,
but it is still model-written code: run it in an environment you are comfortable with.

## Notes

- The ambiguous criteria and their readings were written by the author; they are paraphrased in the style of
  industrial specifications and are not copied from any standard.
- Parts are synthetic pore tables, so disagreement rates depend on the simulated population (see `sensitivity.py`).
- The BDG reference sheet P 202 cited in the paper is not redistributed here; it is available from BDG (www.bdguss.de).

## License

MIT (see `LICENSE`).
