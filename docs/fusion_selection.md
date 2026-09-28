# Fusion weights

The published readout uses `T = 0.45 F + 0.55 S`, `E = 0.25 A + 0.75 S`,
and `O = (T + E) / 2`, after per-subset development-set standardization.
`configs/scoring.json` stores these fixed coefficients.

`contact.select_fusion` implements the development-selection rule. It evaluates
base-head weights from 0 to 1 in increments of 0.05; the silence weight is
one minus the base weight. Timing and affect are selected separately.

Relative to equal-weight fusion, candidates must satisfy these H–H dev constraints:

| Constraint | Timing | Affect |
| --- | ---: | ---: |
| Maximum paired-accuracy decrease | 0.02 | 0.04 |
| Maximum C-index decrease | 0.01 | 0.01 |
| Minimum Combined MOS correlation change | 0 | 0 |

Among admissible candidates, maximize H–AI dev Spearman correlation with
Combined MOS for the corresponding dimension. Ties favor the smaller base-head
weight. Overall keeps equal timing/affect weights; it is not searched.

```bash
python -m contact.select_fusion --data data --output outputs/fusion_selection
```

The script uses only rows marked `dev`. Normalization is fitted on those rows,
separately for each subset, using population standard deviations. Test scores
and ratings do not enter either normalization or selection.

Outputs are `grid.csv` (42 candidates), `selection.json` (rule, selected weights
and a hash of the development inputs), and `normalizers.json`. Use a new output
directory for each run.

This command refits the rule on the supplied development data. The paper
coefficients were retained from the development experiments; a refit on the
released checkpoint scores and split need not recover the same coefficients.
The command never overwrites the published scoring configuration, normalizers,
or evaluation tables.
