# Public validation suites

This directory contains the runnable tests behind the public RTX PRO 6000
results. The historical scores remain evidence from the sealed 2026-08-02
DSpark-4 run. A rerun creates a new output directory and a new result; it does
not overwrite or automatically reproduce those historical numbers.

All runners default to `BASE_URL=http://127.0.0.1:8001` and
`SERVED_MODEL_NAME=pennyroyal`. Both can be set as environment variables or
with command-line options. Use a new `--output-dir` for every live run.

## Validation levels

1. `../scripts/validate-api.sh` checks health, model identity, mapped-W2 audit
   geometry, and a small arithmetic response.
2. `run-reasoning.py` runs the frozen eight-case, nine-request reasoning suite.
3. `run-tools.py` runs the exact 30-invocation tool and agent suite using only
   local mock tool results.
4. `run-needle.py` runs the explicitly acknowledged 994,987-token retrieval
   and immediate arithmetic follow-up.

The collection runners write JSON or JSON Lines manifests and return nonzero
when an automatic gate fails. The reasoning score is different: qualitative
review must be blinded and locked before operational metadata is revealed.

## Frozen reasoning suite

The authoritative cases are in `cases/reasoning.py`, with the original
protocol and rubric beside them. C1–C7 are single-turn. C8 has a first request
and a fixed correction request, producing nine measured requests total. Each
request permits up to 32,768 tokens and intentionally omits request-level
sampling and reasoning overrides so the launcher remains authoritative.

Inspect without contacting a server:

```bash
python3 validation/run-reasoning.py --list
python3 validation/run-reasoning.py --dry-run
python3 validation/run-reasoning.py \
  --replay validation/fixtures/reasoning-replay.jsonl
python3 validation/score-reasoning.py \
  --grade validation/fixtures/reasoning-dspark4-grade.json
```

The final command deterministically reproduces the published **97.07/100**
aggregate from the locked historical grade. It does not regrade prose.

Run a new collection:

```bash
python3 validation/run-reasoning.py \
  --runtime dspark4-new-run \
  --tokenizer-path "$MODEL_PATH" \
  --output-dir validation-results/reasoning-YYYYMMDD-HHMMSS
```

The two warm-ups are not scored. Live collection preserves exact requests,
timestamped SSE lines, reasoning/content fields, usage, finish reasons,
timings, errors, and loop events. Grade response text without runtime, token,
timing, finish, or loop metadata; then use `score-reasoning.py` to calculate
per-case, per-dimension, aggregate, and fatal-capped results. Reviewer judgment
means a nondeterministic grader is not expected to reproduce 97.07 exactly.

Create the text-only blinded packet before grading:

```bash
python3 validation/anonymize-reasoning.py \
  --suite validation/cases/reasoning.py \
  --results validation-results/reasoning-YYYYMMDD-HHMMSS/results.jsonl \
  --output validation-results/reasoning-YYYYMMDD-HHMMSS/grading-packet.json
```

## Tool and agent suite

`cases/tools.json` preserves the exact system prompt, messages, tool schemas,
and deterministic long-context fixture. `cases/tool-expectations.json` records
the exact expected call order and arguments. The only normalization is C13:
the two `delegate_task` calls may be ordered either way, but one task must
identify budget/cost and the other schedule/timeline.

The original invocation schedule is one smoke request, 26 ordinary measured
requests, and three genuinely concurrent requests: **30 invocations** total.
The preserved requests use `reasoning_effort=max`, deterministic per-repeat
seeds, each case's original output cap, and the launcher's sampling defaults.

```bash
python3 validation/run-tools.py --list
python3 validation/run-tools.py --dry-run
python3 validation/run-tools.py \
  --replay validation/fixtures/tools-dspark4-replay.json
python3 validation/run-tools.py \
  --runtime dspark4-new-run \
  --output-dir validation-results/tools-YYYYMMDD-HHMMSS
```

The replay is a minimal, side-effect-free projection of the retained result,
not the sealed raw archive. Live tests return deterministic local mock results
to the model. They never call weather, customer, order, delivery, inventory,
document, restart, delegation, or verification services. The published 30/30
claim means correct parser output, tool selection, and arguments—not execution
against an external system.

### Sealed short performance controls

Two opt-in controls live in the same runner without changing the frozen
30-invocation schedule:

- `sealed_agentic_release_note_v1` performs an ordinary local artifact
  workflow: inspect an authoritative release brief, create a Markdown release
  note, inspect the artifact, revise the reported defect, inspect again, and
  finish with the artifact identity and status. Its exact five-call sequence,
  facts, revision, final inspection, and natural stop are automatically gated.
- `sealed_natural_decode_v2` is the replacement single-stream decode
  instrument. It uses greedy sampling, `reasoning_effort=low`, a 3,072-token
  ceiling, no forced minimum, no `ignore_eos`, and a deliberately boring
  numbered catalog prompt that should reach that ceiling naturally. The result
  records direct server-returned token IDs, a token-ID SHA-256 digest, and an
  independent canonical assistant-output SHA-256 digest.

Both use OpenAI `/v1/chat/completions`, fixed seed 5101, Prometheus counter
deltas, and optional system-journal capture. The agentic control remains
`reasoning_effort=xhigh` with its natural 32,768-token ceiling and
deterministic local tools. Each live invocation requires a unique
`cache_salt`; this isolates the prefix cache without altering the prompt.

Decode v2 supersedes decode v1 from sealed commit
`a557ee12a3ab833165d24b5bd15afb6667d85f81`. Decode v1's unconstrained
content shape produced 3,801 to 9,366 completion tokens and materially
different MTP acceptance, so it was not a stable performance instrument.
The replacement changes benchmark geometry and evidence collection only; it
introduces no runtime or serving optimization.

Inspect either sealed definition without contacting a server:

```bash
python3 validation/run-tools.py --control agentic --dry-run
python3 validation/run-tools.py --control natural-decode --dry-run
```

Run one agentic control and one natural-decode control:

```bash
python3 validation/run-tools.py \
  --control agentic \
  --runtime candidate \
  --cache-key AGENTIC-RUN-0000000000000001 \
  --journal-unit llmbrain \
  --output-dir validation-results/agentic-YYYYMMDD-HHMMSS

python3 validation/run-tools.py \
  --control natural-decode \
  --runtime candidate \
  --cache-key NATURAL-RUN-0000000000000001 \
  --journal-unit llmbrain \
  --output-dir validation-results/natural-YYYYMMDD-HHMMSS
```

The result records total, model, and local-tool wall time separately; model
turns; tool calls; input/output usage; finish reason; effective request rate;
engine decode rate when exposed; speculative draft/accepted tokens and
acceptance rate; prefix-cache queries/hits; bounded journal metrics; and the
parsed journal engine-throughput samples, speculative counts, prefix-hit
samples, and any compile/JIT/CUDA-graph activity observed around the request.
The decode result also records direct token IDs in the raw response and
matching stream digests in both result and manifest, plus the resolved runner
and CUDA graph mode from the current service journal. Use one unmeasured
warm-up and separate output directories for measured repetitions.
Never commit live result directories.

## Opt-in near-million-token needle

The public runner preserves the final filler, system text, retrieval prompt,
needle, streaming collection, and server-rendered token-count authority. Its
default is exactly **994,987 input tokens** with
`GRID-NEEDLE-7B91E2C4A6F0D835` beginning at zero-based token **154**. It reads
DeepSeek output from `reasoning`, `reasoning_content`, and ordinary `content`,
fixing the field mismatch that caused the original generic harness to report a
false negative.

Safe local checks do not contact the model or construct the large prompt:

```bash
python3 validation/run-needle.py --list
python3 validation/run-needle.py --dry-run
python3 validation/run-needle.py --smoke
```

The full run is deliberately impossible without explicit acknowledgement:

```bash
python3 validation/run-needle.py \
  --tokenizer "$MODEL_PATH" \
  --target-input-tokens 994987 \
  --output-dir validation-results/needle-YYYYMMDD-HHMMSS \
  --i-understand-this-may-take-20-minutes
```

The runner requires the endpoint's `/v1/chat/completions/render` result to be
exactly 994,987 tokens, verifies the token position and response, collects
usage, TTFT, prefill/decode rates when `/metrics` exposes them, wall time and
finish reason, then checks that `37 + 58` returns `95`. The historical TTFT was
975.550 seconds—about 16 minutes. This is an exceptional batch workload. Exact
1,048,576-token support was not demonstrated.

## Dependencies and outputs

The tool suite uses the Python standard library. Live reasoning collection
also requires `transformers` for loop detection. The live needle runner
requires `transformers` and `httpx`; its fixture smoke mode does not require a
model or tokenizer. These packages are already present in the documented
native serving environment.

Never commit live result directories. Preserve their requests, raw streams,
responses, manifests, grades, and provenance externally, and assign each run a
new dated identity.

## Provenance

The public material was derived from these retained sources:

| Material | Authoritative SHA-256 |
|---|---|
| Reasoning cases | `d1397529eedf72b0f80d5c452c378ed15fb10f1122e4fb9b50f69e1074c0e756` |
| Reasoning protocol | `56fb8d57dda656495c5b18d5fc6ea35f533bae351d9696cba087fa2913bde928` |
| Reasoning rubric | `ea85f50f9c3cce2f9d9b6b63a3611ffd2e14d9cb90dc4a3d7b5bf7dc0f9c1c65` |
| Reasoning collector | `926db09bef5f3a8a1bde9e24edf506e42687433e233411acb0bc329092707b5b` |
| Reasoning anonymizer | `1d88b2ffe5c7e0938c109d2c50ab253319962f2b1ae7c1529c757b40ceccfffd` |
| Tool runner | `77d5624692e928003500eb1c729bb61ec7aafad6dc0359631d6dca4b25df4299` |
| Tool suite definition | `2ac801193597e444797952f632496de203ac7bf6d1f1b4e5c5eb5de469971bb8` |
| Final needle collector | `63d612f30cd7e2aa4c73a5cd2d897f2ce4a51a636702502f518f5b37bf70ef1a` |

The public versions add portable CLI defaults, replay/dry-run modes, corrected
response-field extraction, strict gates, and the long-run acknowledgement.
Prompts, schemas, rubrics, expected conclusions, and test thresholds are not
tuned to a new run.
