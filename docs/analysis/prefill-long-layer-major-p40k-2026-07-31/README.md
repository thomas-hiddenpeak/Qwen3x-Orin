# P40K layer-major Prefill capacity admission

This host-only admission removes the 4,096-token ceiling from the existing
layer-major executor and raises its checked maximum to 40,960 tokens. It does
not claim GPU correctness or speed.

The motivation is structural. A 40,960-token request contains eighty C512
tiles. Tile-major execution revisits the complete 64-layer weight inventory
for every tile; layer-major execution visits all eighty tiles of one layer
before moving to the next layer. This is required before a compact A4 sidecar
can provide useful long-context weight residency.

The request arena keeps every existing C512 scratch offset and appends two
BF16 `[P,5120]` hidden slabs. The exact admitted capacities are:

| P | Tiles per layer | Work items | Arena bytes |
| ---: | ---: | ---: | ---: |
| 8,192 | 16 | 1,024 | 873,365,504 |
| 16,384 | 32 | 2,048 | 1,580,630,016 |
| 40,960 | 80 | 5,120 | 3,703,209,984 |

For every layer, work items retain ascending global positions. Linear layers
therefore preserve Conv/GDN recurrent-state order, while full-attention layers
append K/V in the same order. The external long-context harness now requires
both the build marker and `Q3X_RUN_LONG_PREFILL_LAYER_MAJOR_ADMISSION=1`, and
rejects a server that does not report a hidden capacity equal to the requested
maximum sequence length.

Synthetic data is used only by host schedule tests. Performance admission
still requires the authorized natural 8K/16K/40K corpora and the real model.

