# DS4 tool-parser lineage

The published DeepSeek-V4 parser combines identifiable merged vLLM behavior
with one narrow MoET adaptation for recursive DSML parameter trees:

| Behavior | Upstream source | Reconciled v0.24 adaptation |
|---|---|---|
| Honor `string="true"` as a literal string and coerce `string="false"` through the declared schema; unwrap a sole artificial `arguments` or `input` object only when its keys fit the real schema | PR [#41801](https://github.com/vllm-project/vllm/pull/41801), merge `95582868efd4db0b120e3640bbc61dcfce20d59f` | Present in the inherited DeepSeek V3.2/V4 parser |
| Stream argument JSON incrementally while buffering split DSML markers | PR [#42879](https://github.com/vllm-project/vllm/pull/42879), merge `b372ad3e9018f032478619adbc7f7fdcc9318212` | Present in the inherited DeepSeek V3.2/V4 parser |
| Move DeepSeek V4 onto the shared streaming parser engine | PR [#45877](https://github.com/vllm-project/vllm/pull/45877), merge `fb5291b35` | Adapted to the MoET v0.24 lineage with the required #46344/#46875 prerequisites |
| Prevent EOS/BOS and other unconfigured special tokens from leaking | PR [#48748](https://github.com/vllm-project/vllm/pull/48748), merge `3de4b2bf3` | Present with the v0.24 test helpers reconciled to the merged scanner design |
| Recover a complete bare invoke only for a tool declared by the current request; reject recovery with no tools or `tool_choice="none"`; preserve rejected and foreign markers as content; reset recovery state between requests | Open PR [#49117](https://github.com/vllm-project/vllm/pull/49117) | Bounded request-scoped recovery retained from Penny's prior parser behavior |
| Decode nested `<｜DSML｜parameter>` elements recursively for open object schemas | MoET v0.24 adaptation | Preserves nested objects, arrays, JSON scalars, empty objects, and incremental streaming without flattening children into the outer call |

The recursive decoder follows the `deepseek_xml` grammar's actual nested DSML
representation. It is not a malformed-output recovery heuristic. Missing outer
or nested fields are not synthesized.

Strict structural grammar is request controlled. This publication does not
force non-strict tools to become strict server-side. An open nested object can
carry arbitrary generated fields, but it cannot require a deferred schema's
unknown fields; the client remains responsible for validating or materializing
that schema.

Focused tests cover streaming and non-streaming extraction, split markers,
guarded wrappers, terminal-style nested schemas, false
orphan matches, foreign wrappers, request-state reset, nested arrays and
objects, literal closing-marker text, and empty open objects. These parser
tests are maintenance checks and do not replace the historical quality suites.
