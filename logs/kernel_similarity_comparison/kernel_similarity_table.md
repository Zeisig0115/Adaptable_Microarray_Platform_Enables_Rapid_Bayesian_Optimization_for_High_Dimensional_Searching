# Kernel Similarity Comparison

- Data: `C:\PyCharm\Blueness\logs\May_10_full_log\data_corrected.xlsx`
- Sheet/filter: `Corrected Data`, `HRP=0.0001`, `CTRL=0`
- Similarity: normalized covariance `K(x, y) / sqrt(K(x, x) K(y, y))` after each model's input transform
- Fit: imported `add_bo` / `add_bo_mod` builders, `fit_maxiter=75`, no candidates

| Pair | Meaning | OLD | NEW | MOD |
| --- | --- | --- | --- | --- |
| r460 blank vs r461 blank | Blank vs blank; different TMB/H2O2 | 0.85 | 1.00 | 0.99 |
| r000 cmc=1 vs r277 cmc=1 | Same additive and dose; A/B replicate recipe | 1.00 | 1.00 | 1.00 |
| r276 cmc=2 vs r280 cmc=0.125 | Same additive; different CMC dose | 0.99 | 0.97 | 1.00 |
| r000 cmc=1 vs r105 pva=0.0078125 | Same family polymer; CMC vs PVA | 0.87 | 0.84 | 0.73 |
| r270 peg6k=1 vs r273 peg400=1 | Same family PEG; PEG6k vs PEG400 | 0.85 | 0.84 | 0.70 |
| r231 cacl2=0.25 vs r266 feso4=1 | Same family salt; CaCl2 vs FeSO4 | 0.33 | 0.84 | 0.68 |
| r000 cmc=1 vs r045 dmso=1 | Different family; polymer vs solvent | 0.82 | 0.84 | 0.43 |
| r231 cacl2=0.25 vs r045 dmso=1 | Different family; salt vs solvent | 0.34 | 0.84 | 0.61 |
| r000 cmc=1 vs r273 peg400=1 | Disjoint additives; both polymers | 0.85 | 0.84 | 0.58 |
| r460 blank vs r000 cmc=1 | Blank vs single additive | 0.58 | 0.85 | 0.12 |
| r278 cmc=0.5 vs r021 cmc=0.5+peg400=0.5 | Subset relation; CMC vs CMC+PEG400 | 0.93 | 0.92 | 0.89 |

## Fit Metadata

```json
{
  "encode_meta": {
    "raw_rows": 461,
    "unique_encoded_rows": 446,
    "duplicates_merged": 15,
    "encoded_dim": 46,
    "n_additives": 22
  },
  "fit_info": {
    "old_mixed_hamming": {
      "fit_seconds": 14.581124782562256,
      "mll": -0.5958555025189316
    },
    "new_additive_set": {
      "fit_seconds": 10.772573947906494,
      "mll": -0.21856696671950882
    },
    "hierarchical_family_prior": {
      "mll": -0.04570130687951313
    }
  }
}
```
