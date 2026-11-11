# Honest Error Analysis (`DubaiFairValue`)

Model performance across segments, temporal drift over test months, and worst test predictions.

## Segment Error & Interval Coverage Breakdown

| Segment | Test Rows | MdAPE | 90% Interval Coverage |
| --- | ---: | ---: | ---: |
| All Test Transactions | 70,608 | 6.94% | 89.9% |
| Apartments (Unit) | 62,613 | 6.41% | 90.4% |
| Villas & Townhouses | 7,995 | 13.53% | 86.0% |
| Off-Plan Sales | 48,345 | 6.29% | 89.4% |
| Ready / Secondary Sales | 22,263 | 8.68% | 91.0% |
| Luxury (>= AED 5M) | 3,231 | 13.18% | 81.0% |
| Thin-Data Areas (<150 trailing sales) | 1,077 | 7.42% | 92.9% |

## Monthly Drift Across Test Window

| Month | Rows | MdAPE | Coverage |
| --- | ---: | ---: | ---: |
| 2024-02 | 2,332 | 6.52% | 90.9% |
| 2024-03 | 10,595 | 6.63% | 90.0% |
| 2024-04 | 9,519 | 6.59% | 90.9% |
| 2024-05 | 14,319 | 6.70% | 90.2% |
| 2024-06 | 10,858 | 7.05% | 90.2% |
| 2024-07 | 13,101 | 7.07% | 89.9% |
| 2024-08 | 9,884 | 7.74% | 87.9% |

## Top 10 Worst Test Predictions (with Likely Causes)

| Date | Area | Type | Area (sqm) | Actual (AED) | Predicted (AED) | Abs % Error | Note |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| 2024-05-20 | Dubai Marina | Unit | 410 | 1,549,888 | 11,807,190 | 661.8% | unobserved view/floor/fit-out |
| 2024-03-28 | Dubai Marina | Unit | 178 | 1,000,000 | 6,099,323 | 509.9% | off-plan payment/launch tier |
| 2024-07-29 | Dubai Marina | Unit | 33 | 180,395 | 847,855 | 370.0% | unobserved view/floor/fit-out |
| 2024-04-02 | Dubai Marina | Unit | 86 | 741,440 | 2,628,182 | 254.5% | off-plan payment/launch tier |
| 2024-04-02 | Dubai Marina | Unit | 85 | 730,240 | 2,587,423 | 254.3% | off-plan payment/launch tier |
| 2024-08-15 | Dubai Marina | Unit | 136 | 592,884 | 1,955,876 | 229.9% | off-plan payment/launch tier |
| 2024-08-15 | Dubai Marina | Unit | 136 | 595,080 | 1,955,876 | 228.7% | off-plan payment/launch tier |
| 2024-08-15 | Dubai Marina | Unit | 129 | 561,624 | 1,845,261 | 228.6% | off-plan payment/launch tier |
| 2024-03-04 | Palm Jumeirah | Unit | 188 | 2,500,000 | 8,054,520 | 222.2% | off-plan payment/launch tier |
| 2024-08-13 | Dubai Marina | Unit | 136 | 613,548 | 1,967,950 | 220.7% | off-plan payment/launch tier |
