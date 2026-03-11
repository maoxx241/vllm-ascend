# Matrix Heuristics

## Good Control Dimensions

- graph vs eager
- profiling on vs off
- one batch-size change
- one request-shape change
- one parallelism knob
- one scheduling knob
- one quantization or cache knob

## Typical Minimal Matrices

### Graph suspicion

- baseline graph-on
- eager fallback
- graph-on with reduced shape variance

### Scheduler suspicion

- baseline request mix
- lower concurrency
- changed prefill/decode balance

### Memory pressure suspicion

- baseline
- lower `max-num-seqs`
- lower `max-model-len` or altered KV-related knob

## Required Per-Run Outputs

- command line and env diff
- wall-clock summary
- one profiling summary artifact
- one note about whether the symptom reproduced
