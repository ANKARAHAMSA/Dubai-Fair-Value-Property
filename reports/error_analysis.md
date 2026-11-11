# Honest Error Analysis (`DubaiFairValue`)

Model performance across segments, temporal drift over test months, and worst test predictions.

## Segment Error & Interval Coverage Breakdown

| Segment | Test Rows | MdAPE | 90% Interval Coverage |
| --- | ---: | ---: | ---: |
| All Test Transactions | 70,608 | 6.89% | 90.1% |
| Apartments (Unit) | 62,613 | 6.41% | 90.5% |
| Villas & Townhouses | 7,995 | 12.52% | 87.0% |
| Off-Plan Sales | 48,345 | 6.19% | 89.5% |
| Ready / Secondary Sales | 22,263 | 8.74% | 91.4% |
| Luxury (>= AED 5M) | 3,231 | 13.39% | 81.9% |
| Thin-Data Areas (<150 trailing sales) | 1,077 | 7.94% | 91.7% |

## Monthly Drift Across Test Window

| Month | Rows | MdAPE | Coverage |
| --- | ---: | ---: | ---: |
| 2024-02 | 2,332 | 6.57% | 90.1% |
| 2024-03 | 10,595 | 6.74% | 90.1% |
| 2024-04 | 9,519 | 6.32% | 90.8% |
| 2024-05 | 14,319 | 6.67% | 91.0% |
| 2024-06 | 10,858 | 6.94% | 90.8% |
| 2024-07 | 13,101 | 7.07% | 89.7% |
| 2024-08 | 9,884 | 7.63% | 88.0% |

## Top 10 Worst Test Predictions (with Likely Causes)

| Date | Area | Type | Area (sqm) | Actual (AED) | Predicted (AED) | Abs % Error | Note |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| 2024-05-20 | Dubai Marina | Unit | 410 | 1,549,888 | 11,394,805 | 635.2% | unobserved view/floor/fit-out |
| 2024-03-28 | Dubai Marina | Unit | 178 | 1,000,000 | 6,128,553 | 512.9% | off-plan payment/launch tier |
| 2024-07-29 | Dubai Marina | Unit | 33 | 180,395 | 841,252 | 366.3% | unobserved view/floor/fit-out |
| 2024-04-02 | Dubai Marina | Unit | 85 | 730,240 | 2,443,700 | 234.6% | off-plan payment/launch tier |
| 2024-04-02 | Dubai Marina | Unit | 86 | 741,440 | 2,481,154 | 234.6% | off-plan payment/launch tier |
| 2024-03-04 | Palm Jumeirah | Unit | 188 | 2,500,000 | 8,317,800 | 232.7% | off-plan payment/launch tier |
| 2024-06-11 | Dubai Marina | Unit | 99 | 500,000 | 1,641,582 | 228.3% | unobserved view/floor/fit-out |
| 2024-08-15 | Dubai Marina | Unit | 136 | 592,884 | 1,901,459 | 220.7% | off-plan payment/launch tier |
| 2024-08-15 | Dubai Marina | Unit | 129 | 561,624 | 1,798,064 | 220.2% | off-plan payment/launch tier |
| 2024-08-15 | Dubai Marina | Unit | 136 | 595,080 | 1,901,459 | 219.5% | off-plan payment/launch tier |
