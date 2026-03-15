# Evidence priority

Use evidence in this order:

1. local observable facts
   - config.json
   - model path
   - repo source
   - local upstream mirror
   - actual error messages
2. repo docs and tutorials
3. nightly / test configs
4. support matrix
5. user-only facts
6. lightweight numerical checks such as resource lower bounds

Absence of a higher-priority positive signal is not the same as a negative signal.
