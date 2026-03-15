# glm46.md Analysis

## Scope

This note summarizes why the deployment bundle generated from the remote transcript at
`/workspace/vllm_workspace/vllm-ascend/glm46.md`
used `tp16` instead of a `dp+tp` split, and why it did not enable `ep`.

## Conclusion

The generated script was not the result of a GLM-aware topology solver. It was effectively derived from the documented single-node low-latency recommendation in the repo tutorial, then written back by Kimi as if it were a deployment bundle.

Two direct consequences followed:

- `tp16` was selected because the tutorial explicitly recommends `dp1tp16` for single-node deployment.
- `ep` was not enabled because the same tutorial explicitly recommends turning off expert parallel for that low-latency single-node path.

## Why `tp16`

The strongest evidence is the GLM tutorial itself:

- [docs/source/tutorials/models/GLM4.x.md](/Users/maoxx241/code/vllm_workspace/vllm-ascend/docs/source/tutorials/models/GLM4.x.md#L76)

That section contains a concrete single-node command with:

- `--data-parallel-size 1`
- `--tensor-parallel-size 16`

and it explicitly states:

- for single-node deployment, recommend `dp1tp16`

So the transcript output matched the tutorial recommendation directly rather than synthesizing a topology from a broader evidence set.

## Why no `ep`

The same tutorial section also states that in the single-node low-latency scenario expert parallel should be turned off:

- [docs/source/tutorials/models/GLM4.x.md](/Users/maoxx241/code/vllm_workspace/vllm-ascend/docs/source/tutorials/models/GLM4.x.md#L107)

That means the generated bundle was following a documented low-latency recipe, not performing a fresh MoE strategy decision.

## Why this was not a real open-world deployment synthesis result

There is repo evidence that a `dp+tp` route exists:

- [tests/e2e/nightly/single_node/models/configs/GLM-4.5.yaml](/Users/maoxx241/code/vllm_workspace/vllm-ascend/tests/e2e/nightly/single_node/models/configs/GLM-4.5.yaml#L55)

That nightly config includes:

- `GLM-4.5-TP8-DP2-fullgraph`
- `GLM-4.5-TP8-DP2-eager`

So the repo does contain a `TP8 + DP2` shape, but the transcript did not use it as the final deployment decision.

The current accepted deployment implementation also does not have GLM-specific reasoning:

- [tools/vas_deployment_skill/parser.py](/Users/maoxx241/code/vllm_workspace/vllm-ascend/tools/vas_deployment_skill/parser.py#L9)
- [tools/vas_deployment_skill/engine.py](/Users/maoxx241/code/vllm_workspace/vllm-ascend/tools/vas_deployment_skill/engine.py#L37)
- [tools/vas_deployment_skill/engine.py](/Users/maoxx241/code/vllm_workspace/vllm-ascend/tools/vas_deployment_skill/engine.py#L77)

Observed gaps:

- `parser.py` does not recognize `glm` / `glm4.6`
- `engine.py` has no GLM-specific evidence collection branch
- `engine.py` has no GLM-specific deployment candidate branch

Because of that, if the accepted deployment engine had actually been used end-to-end, it would not have produced a confident GLM-specific `exact_verified` bundle with `tp16` and `ep=false`.

## Final Assessment

`glm46.md` should be read as a doc-driven manual answer, not as proof that the accepted deployment runtime can currently reason about GLM topology.

The practical interpretation is:

- `tp16` came from the tutorial's single-node low-latency recommendation
- `ep` stayed off for the same reason
- the transcript did not demonstrate real GLM-specific open-world deployment synthesis
