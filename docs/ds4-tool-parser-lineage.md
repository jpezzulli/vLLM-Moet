# DS4 tool-parser lineage

The published DeepSeek-V4 parser patch is intentionally limited to identifiable
upstream vLLM behavior:

| Behavior | Upstream source | Reconciled v0.24 adaptation |
|---|---|---|
| Honor `string="true"` as a literal string and coerce `string="false"` through the declared schema; unwrap a sole artificial `arguments` or `input` object only when its keys fit the real schema | PR [#41801](https://github.com/vllm-project/vllm/pull/41801), merge `95582868efd4db0b120e3640bbc61dcfce20d59f` | Present in the inherited DeepSeek V3.2/V4 parser |
| Stream argument JSON incrementally while buffering split DSML markers | PR [#42879](https://github.com/vllm-project/vllm/pull/42879), merge `b372ad3e9018f032478619adbc7f7fdcc9318212` | Present in the inherited DeepSeek V3.2/V4 parser |
| Recover a complete bare invoke only for a tool declared by the current request; reject recovery with no tools or `tool_choice="none"`; preserve rejected and foreign markers as content; reset recovery state between requests | Open PR [#49117](https://github.com/vllm-project/vllm/pull/49117), head `7ef0ae2480799e95fb7cb801a8105c1db2585164` inspected August 9, 2026 | The newer shared parser-engine transitions are represented by equivalent request-scoped state in the v0.24 tool parser |

Nested objects and arrays are represented as JSON text inside a
`string="false"` top-level parameter. The publication does not carry the
previous local recursive nested-DSML interpretation or a malformed-output
repair heuristic.

Focused tests cover streaming and non-streaming extraction, split markers,
guarded wrappers, terminal-style nested schemas, false
orphan matches, foreign wrappers, and request-state reset. These parser tests
are maintenance checks and do not replace the historical quality suites.
