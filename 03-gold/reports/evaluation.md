# Model evaluation

- Model configuration: C=1.0, parent_weight=1.0
- Temperature: 0.986228
- ECE before -> after calibration: 0.025894 -> 0.023766

## Cross-validation (per fold)

| Fold | Accuracy | Macro F1 | Log loss |
| ---: | ---: | ---: | ---: |
| 0 | 0.779201 | 0.710132 | 0.74202 |
| 1 | 0.780096 | 0.709794 | 0.728381 |
| 2 | 0.790096 | 0.719773 | 0.719536 |
| 3 | 0.784681 | 0.717653 | 0.757414 |
| 4 | 0.792878 | 0.721928 | 0.719194 |
| mean | 0.78539 | 0.715856 | 0.733309 |

## Baseline comparison

| Model | Pooled OOF log loss | Mean accuracy | Mean macro F1 |
| --- | ---: | ---: | ---: |
| logistic regression | 0.733401 | 0.78539 | 0.715856 |
| naive bayes | 1.285516 | 0.743931 | 0.657276 |

## Per-cuisine recall

| Cuisine | Recipes | Recall |
| --- | ---: | ---: |
| brazilian | 467 | 0.588865 |
| british | 804 | 0.437811 |
| cajun_creole | 1546 | 0.69599 |
| chinese | 2673 | 0.83988 |
| filipino | 755 | 0.637086 |
| french | 2646 | 0.626228 |
| greek | 1175 | 0.690213 |
| indian | 3003 | 0.889111 |
| irish | 667 | 0.511244 |
| italian | 7838 | 0.884792 |
| jamaican | 526 | 0.68251 |
| japanese | 1423 | 0.708363 |
| korean | 830 | 0.789157 |
| mexican | 6438 | 0.91923 |
| moroccan | 821 | 0.751523 |
| russian | 489 | 0.458078 |
| southern_us | 4320 | 0.80162 |
| spanish | 989 | 0.489383 |
| thai | 1539 | 0.764133 |
| vietnamese | 825 | 0.595152 |

## Top confusion pairs (vs taxonomy neighbors)

| True | Predicted | Count | Neighbor similarity |
| --- | --- | ---: | ---: |
| french | italian | 533 | 0.6621 |
| italian | french | 362 | 0.6621 |
| cajun_creole | southern_us | 261 | 0.5354 |
| southern_us | italian | 196 | — |
| greek | italian | 195 | 0.5677 |
| spanish | italian | 185 | 0.6589 |
| french | southern_us | 181 | 0.6604 |
| italian | southern_us | 179 | 0.473 |
| southern_us | cajun_creole | 159 | — |
| vietnamese | thai | 154 | 0.8128 |
| mexican | southern_us | 145 | — |
| southern_us | french | 139 | 0.6604 |
| japanese | chinese | 135 | 0.6251 |
| british | french | 131 | 0.7205 |
| southern_us | mexican | 125 | — |
