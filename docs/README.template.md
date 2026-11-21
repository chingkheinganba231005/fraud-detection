# Medicare Provider Fraud Detection

Which doctors and clinics billing Medicare will be excluded for fraud in the next two years? This project ranks about 1.2 million US providers from their public billing records and checks the ranking against the federal exclusion list.

On 2022 providers the model never saw, a provider later excluded for fraud is ranked above a provider who was not in {{auc_pct}} of pairs (ROC AUC {{auc}}, 95% CI {{auc_ci}}), at a base rate of about 1 in 15,000. An investigation team reviewing the top 5% of providers would find {{hits5}} of the {{test_positives}} providers that the Office of Inspector General excluded for fraud in 2023 and 2024 ({{recall5}}, 95% CI {{recall5_ci}}). That is {{lift5}} times as many as a random list of the same size. Medicare paid those {{hits5}} providers {{value5}} in 2022, against {{value5_random}} expected from a random list.

## The problem

Insurers lose money to providers who bill for care that was never given, bill a cheap service as an expensive one, or take kickbacks for referrals. A fraud team can only look at a small share of providers, so the useful question for a model is which providers to look at first, and how many real cases a list of a given length would contain.

Real fraud labels are rarely public. The closest public substitute is the list of providers the US Office of Inspector General (OIG) has excluded from federal health programmes. This project joins that list to the billing each provider did before the exclusion and asks whether the billing alone predicts it.

## Data

| Source | What it has | Years |
|---|---|---|
| CMS Medicare Physician and Other Practitioners, by provider (Part B) | patients, services, charges, payments and patient mix for each provider | 2014–2017, 2019, 2022 |
| CMS Medicare Part D Prescribers, by provider | prescriptions, drug cost, and brand, opioid, antipsychotic and low-income-subsidy shares | same years |
| OIG List of Excluded Individuals and Entities (LEIE) | exclusion type and date for every excluded provider with an NPI | current list |

That is {{all_rows}} provider-years. The URL, size, SHA-256 and download date of every file are in [`results/data_manifest.json`](results/data_manifest.json).

## Labels

A provider-year is positive if the provider was excluded for a fraud-related reason in the two calendar years after it. The fraud-related authorities are listed in [`src/fraud/config.py`](src/fraud/config.py): program-related convictions, health care fraud convictions, kickbacks, excessive or unnecessary services, obstruction and civil monetary penalties. Providers already excluded before or during the billing year are dropped ({{dropped_prior}} provider-years), and so are providers excluded only for other reasons, such as losing a licence ({{dropped_other}}).

Every other provider is a negative, which really means "not caught". Undetected fraud is among the negatives, so the precision figures below are lower bounds.

| Split | Billing years | Exclusions counted | Provider-years | Positives |
|---|---|---|---|---|
| training | {{train_years}} | 2015–2019 | {{train_rows}} | {{train_positives}} ({{train_positive_providers}} providers) |
| validation | 2019 | 2020–2021 | {{val_rows}} | {{val_positives}} |
| test | 2022 | 2023–2024 | {{test_providers}} | {{test_positives}} |

The exclusion windows do not overlap, so no exclusion is used both to fit a model and to judge it.

## Method

All features are built in DuckDB SQL ([`sql/`](sql/)) from one row per provider and year: volumes on a log scale, and the ratios an investigator checks first, such as services per patient, payment per service, charges relative to the allowed amount, the share of patients also on Medicaid, cost per prescription and the share of opioid prescriptions. Each ratio is also expressed against providers of the same specialty in the same year, as the distance from the specialty median in interquartile ranges. Names, addresses, NPIs and the state are never model inputs.

Six detectors are compared:

| Detector | What it does |
|---|---|
| peer rule | the largest upward deviation from the specialty median on six billing ratios, the classic outlier screen |
| isolation forest | unsupervised anomaly score on all features |
| logistic regression | class-weighted, on standardised features |
| LightGBM | gradient boosting with early stopping on the validation year |
| nearest known cases | similarity to the {{indexed_cases}} providers excluded in the training years |
| LightGBM + peer rule | average of the two percentile ranks |

The rule for choosing one was fixed before the test year was loaded: the most later-excluded providers in the top 1,000 of 2019, with average precision as the tie-break. The test year was scored once, with every detector.

## Results

Validation (2019):

{{validation_table}}

Test (2022), list of {{budget}}:

{{detector_table}}

A list of {{budget}} is only {{share_reviewed}} of all providers, and at this base rate a random list of that length would contain {{random_hits}} later-excluded providers. The model's list contains {{hits}}, which is too few to say much. Longer lists, as a share of all providers, are more informative. These shares were added after the first test run, and the model was not changed.

Chosen detector ({{chosen}}) on the test year, with 95% bootstrap intervals from resampling the positives:

{{budget_table}}

![Recall by review budget](figures/recall_by_budget.png)

Among the providers found in the top 5%, the most common specialties were {{hit_specialties}}. They were on the list a median of {{lead_median}} months before their exclusion (range {{lead_min}} to {{lead_max}} months), counted from the end of 2022. Logistic regression does about as well as LightGBM on the longer lists, so most of the signal is simple enough for a linear model.

What the model relies on is worth stating plainly. For {{reason_specialty}} of the top {{budget}} providers, the largest single contribution to the score is the specialty, and the next most important inputs are the patient mix (the shares of patients on Medicaid or receiving the low-income drug subsidy) and the mark-up of charges over the allowed amount. The peer rule is weak on the real exclusions (ROC AUC {{auc_peer}}). Providers later excluded for fraud mostly do not stand out as extreme billers in yearly totals. Most of these exclusions follow criminal convictions, often for kickbacks or schemes that may not show up in yearly totals at all. Organisations make up {{org_pop}} of providers and {{org_list}} of the top-{{budget}} list. Rural providers make up {{rural_pop}} of providers and {{rural_list}} of the list.

![Feature importance](figures/importance.png)

## Synthetic stress test

The labels only cover fraud that was caught, prosecuted and ended in an exclusion. To see whether the detectors would notice schemes that are common but rarely prosecuted, {{stress_n}} real 2022 providers who were never excluded are copied, one scheme is applied to the copy's billing, and the copy is scored against the clean 2022 population. A copy counts as detected if it would make the top 1% list ({{stress_list}} providers). The copies are never used for training or for the results above.

| Scheme | What changes |
|---|---|
| phantom services | services and payments go up, patients stay the same |
| upcoding | payment per service goes up, services stay the same |
| patient mill | patients, services and payments all go up together, every ratio stays the same |
| opioid prescribing | extra opioid prescriptions, at the provider's own cost per prescription |

{{stress_table}}

![Stress test](figures/stress_test.png)

{{stress_summary}}

## A bug the stress test found

The first full run had a bug in the peer z-scores. DuckDB's `least()` and `greatest()` skip NULLs, so clipping a missing value to [-50, 50] returned 50. Every provider without Part D data, about a third of them, looked like an extreme outlier on every prescription metric. The model still ranked reasonably (ROC AUC {{first_auc}} on 2022, {{first_hits5}} later-excluded providers in the top 5%), because a constant 50 works like a missing-value flag. The peer rule was badly hurt (ROC AUC {{first_peer_auc}}). The stress test exposed the bug: unchanged copies of real providers were flagged far above the background rate.

Missing values now stay missing, peer groups with an IQR of zero use the standard deviation, and two tests cover both cases. The models were refitted and chosen again on 2019 with the same rule, and 2022 was scored a second time. Both test runs are in `results/` (`first_run_test.json` and `test.json`), and every number above comes from the second.

## Scoring service

```bash
make serve
curl -s localhost:8000/score -H 'Content-Type: application/json' -d @tests/example_provider.json
```

The service takes one provider's yearly Part B totals (and Part D totals if there are any), builds the features with the same SQL as the pipeline, and returns the score, its percentile among 2022 providers, whether it would be on the top-{{budget}} list, the three inputs that raised the score most in plain language, and the five most similar providers that were later excluded, with their specialty and exclusion type. On its own, similarity to known cases ranks weakly (ROC AUC {{auc_knn}} on the test year), so it is shown to the reviewer as context and is not part of the score.

## Limitations

- An exclusion is not the same as fraud. Most fraud is never caught, and an exclusion usually follows a conviction, often years after the billing.
- The LEIE is the current list. Providers who were reinstated have dropped off it, and exclusions without an NPI cannot be linked to claims.
- Yearly totals per provider hide the claim-level patterns an insurer would see, such as codes billed together, dates, diagnoses and referral networks.
- CMS publishes each year's data about 18 months after the year ends, so the lead time above is counted from the end of the billing year. An insurer scoring its own claims could flag providers much sooner.
- With {{test_positives}} positives in the test year, the intervals are wide.

## Reproduce

```bash
make install
make all     # download about 6 GB, build the dataset, train, evaluate, stress test, README
make test    # pytest on a small real sample in tests/data
```

`make all` takes about an hour on 4 cores. Seeds are fixed in `src/fraud/config.py`, and every number in this README is filled in from `results/*.json` by `scripts/render_readme.py`. CI fails if the README and the results disagree.

| Path | Contents |
|---|---|
| `sql/` | load, labels, provider-year ratios, peer statistics, peer z-scores |
| `src/fraud/` | detectors, evaluation, stress test schemes, scorer, service |
| `scripts/` | the pipeline steps, in order |
| `models/` | the chosen LightGBM model, baselines, case index and reference files |
| `results/`, `figures/` | everything the README reports |
| `tests/` | tests and the real data sample |
