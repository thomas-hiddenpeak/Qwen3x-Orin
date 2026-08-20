# Qwen3.6-27B-NVFP4 per-block zero statistics

- Source: `/home/rm01/models/dev/llm/nvidia/Qwen3.6-27B-NVFP4`
- Revision: `0893e1606ff3d5f97a441f405d5fc541a6bdf404`
- Quantization block: one output row × 16 consecutive K values
- NVFP4 tensors: 193
- Blocks: 1,149,009,920
- Logical NVFP4 values: 18,384,158,720
- Scan time: 159.78 seconds

## Headline

- Strict dequantized zeros: 1,461,170,821 (7.9479885006%)
- E2M1 +0 codes: 730,618,492
- E2M1 -0 codes: 730,552,329
- Blocks containing at least one zero: 829,302,967 (72.1754401389%)
- Fully zero blocks: 0
- Positive, finite block scales: 1,149,009,920 / 1,149,009,920

All block scales and tensor-level scales are positive and finite, so strict dequantized zero equals E2M1 code 0x0/0x8 for this checkpoint.

## By projection

| Projection | Blocks | Zero values | Zero rate | Any-zero blocks | Fully zero |
|---|---:|---:|---:|---:|---:|
| gate | 356,515,840 | 451,537,249 | 7.91579921% | 72.04139766% | 0 |
| up | 356,515,840 | 454,170,694 | 7.96196555% | 72.26936929% | 0 |
| down | 356,515,840 | 453,993,356 | 7.95885668% | 72.21092561% | 0 |
| lm_head | 79,462,400 | 101,469,522 | 7.98093831% | 72.19620349% | 0 |

## Per-block zero-count histogram

| Zeros in 16-value block | Blocks | Share |
|---:|---:|---:|
| 0 | 319,706,953 | 27.8245598611% |
| 1 | 411,085,902 | 35.7774023396% |
| 2 | 262,556,728 | 22.8506928817% |
| 3 | 110,689,449 | 9.6334633038% |
| 4 | 34,483,357 | 3.0011365785% |
| 5 | 8,435,736 | 0.7341743403% |
| 6 | 1,687,484 | 0.1468641803% |
| 7 | 291,939 | 0.0254078746% |
| 8 | 50,410 | 0.0043872554% |
| 9 | 11,927 | 0.0010380241% |
| 10 | 4,611 | 0.0004013020% |
| 11 | 2,329 | 0.0002026962% |
| 12 | 1,355 | 0.0001179276% |
| 13 | 826 | 0.0000718880% |
| 14 | 529 | 0.0000460396% |
| 15 | 385 | 0.0000335071% |
| 16 | 0 | 0.0000000000% |

## Transformer-layer range

| Rank | Layer | Zero rate | Any-zero blocks |
|---:|---|---:|---:|
| high 1 | layer_63 | 8.15664703% | 72.98916087% |
| high 2 | layer_34 | 8.14800636% | 72.89328781% |
| high 3 | layer_50 | 8.14069860% | 72.90107278% |
| high 4 | layer_33 | 8.05263370% | 72.58214016% |
| high 5 | layer_49 | 8.04197050% | 72.50590007% |
| low 1 | layer_03 | 7.84228605% | 71.74233231% |
| low 2 | layer_08 | 7.85600400% | 71.84539197% |
| low 3 | layer_09 | 7.85636004% | 71.81482053% |
| low 4 | layer_02 | 7.85637313% | 71.78824032% |
| low 5 | layer_12 | 7.85743900% | 71.82131898% |

## Files

- `summary.csv`: overall, projection, layer, and tensor summaries.
- `block_zero_histogram.csv`: zero-count histograms at all scopes.
- `e2m1_code_histogram.csv`: packed E2M1 code distributions.
- `stats.json`: full machine-readable result and provenance.

The raw 1,149,009,920-row block table is intentionally not emitted; a uint8 zero count per block would still be about 1.15 GB.
