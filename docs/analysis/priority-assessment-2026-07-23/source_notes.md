# Source and report notes

## Decision frame

- Decision: whether to continue the recent batch-one Decode micro-optimization
  sequence or first insert the proposed Prefill/Decode architecture split.
- Audience: technical project owner.
- Snapshot: repository HEAD `06b1a52` and local profiling artifacts whose byte
  sizes and SHA-256 identities are recorded by that committed evidence, reviewed
  on 2026-07-23 Asia/Shanghai.
- Recommended choice: a bridge strategy—close and freeze the current baseline,
  implement a no-behavior-change phase seam, then continue the original Phase 3
  large-Prefill/dispatch work through that seam.

## Evidence hierarchy and quality

1. `docs/ROADMAP.md` and `docs/PERFORMANCE_BASELINE.md` define the planned phase
   dependency and current project decision.
2. Committed benchmark JSON records control current and historical measured
   values. The latest end-to-end data is unlocked-clock B-C-C-B diagnostic
   evidence with two process medians per side; use it directionally, not as a
   release claim.
3. The C16/C8 comparison is same-binary, mirrored order, eight measured samples
   per policy, but covers two short prompts and two output tokens.
4. The latest Nsight breakdown is one exact max-26 generation. Its local CSV and
   profile log were re-read and their hashes match the committed metadata. Phase
   shares are diagnostic and do not establish large-prompt or serving behavior.

## Report structure mapping

- Title: `title` block.
- Technical summary: `technical-summary` block.
- Key findings with evidence: `evidence-heading`, metric strip, and
  `evidence-table`.
- Scope/data/metric definitions: `scope-definitions`.
- Methodology: `methodology` plus `strategy-table`.
- Limitations and robustness: `limitations`.
- Recommended next steps: `next-steps` plus `priority-table`.
- Further questions: `further-questions`.

## Chart map

- Section: Decode hotspot concentration.
- Question: how concentrated is current max-26 CUDA time in the largest M1
  kernel families?
- Family/type: comparison and ranking / horizontal bar.
- Fields: kernel group, exact total milliseconds, share of all CUDA kernel time,
  instance count, rank, and phase.
- Takeaway: the six largest M1 groups account for 83.94% of CUDA kernel time and
  are already specialized or fused, supporting a move away from tiny Decode
  micro-adjustments.
- Palette: single blue root, direct value labels, no redundant legend.
- Surface: native chart in the portable HTML report.

No historical trend chart is used. The evidence mixes a current snapshot, a
same-binary C8/C16 experiment, and cross-commit historical context. Plotting
those as a trend would imply comparability that the sources explicitly reject.
Exact tables remain the more faithful format for those comparisons.

## Reproducibility

`priority_analysis.ipynb` loads the committed benchmark JSON and the bounded
`evidence.csv`/`hotspots.csv` extracts, recomputes the decision-driving ratios,
and checks that the derived values reconcile. The local Nsight CSV used to derive the bounded rows
has SHA-256 `882f1c452c91...bdd813f2777929ddefa05b7d5f74faa1`, matching the
committed metadata record.

The notebook completed with no cell errors and independently reconciles TTFT,
Decode throughput, the wall/kernel envelope, the C16/C8 result, and the exact
83.939275% top-six M1 share. Portable-report validation and packaging passed.
Browser-level visual and source-dialog interaction checks were not available
because this host has no installed Chromium; delivery is therefore structurally
verified rather than screenshot-verified.
