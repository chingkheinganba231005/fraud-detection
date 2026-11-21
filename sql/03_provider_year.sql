-- One row per provider and year: volumes on a log scale and the ratios investigators look
-- at (services per patient, payment per service, mark-up over the allowed amount, drug mix).
-- Reads the tables partb and partd, so the same query serves the batch pipeline, the
-- synthetic stress test and the scoring service.

CREATE OR REPLACE TABLE provider_year AS
SELECT
    b.year,
    b.npi,
    coalesce(b.provider_type, 'Unknown')                        AS provider_type,
    b.entity,
    b.state,
    b.ruca,
    CASE WHEN b.participating = 'Y' THEN 1 ELSE 0 END           AS participating,
    b.b_payment,
    d.d_cost,
    ln(1 + b.b_benes)                                           AS log_b_benes,
    ln(1 + b.b_services)                                        AS log_b_services,
    ln(1 + b.b_payment)                                         AS log_b_payment,
    ln(1 + b.b_codes)                                           AS log_b_codes,
    b.b_services / nullif(b.b_benes, 0)                         AS services_per_bene,
    b.b_payment / nullif(b.b_benes, 0)                          AS payment_per_bene,
    b.b_payment / nullif(b.b_services, 0)                       AS payment_per_service,
    b.b_charges / nullif(b.b_allowed, 0)                        AS charge_to_allowed,
    b.b_codes / nullif(b.b_benes, 0)                            AS codes_per_bene,
    coalesce(b.b_drug_payment, 0) / nullif(b.b_payment, 0)      AS drug_payment_share,
    b.bene_age,
    b.bene_lt65 / nullif(b.b_benes, 0)                          AS share_under_65,
    b.bene_gt84 / nullif(b.b_benes, 0)                          AS share_over_84,
    b.bene_female / nullif(b.b_benes, 0)                        AS share_female,
    b.bene_dual / nullif(b.b_benes, 0)                          AS share_dual,
    b.risk_score,
    CASE WHEN d.d_claims IS NULL THEN 0 ELSE 1 END              AS has_part_d,
    ln(1 + d.d_claims)                                          AS log_d_claims,
    ln(1 + d.d_cost)                                            AS log_d_cost,
    d.d_cost / nullif(d.d_claims, 0)                            AS d_cost_per_claim,
    d.d_days / nullif(d.d_claims, 0)                            AS d_days_per_claim,
    d.d_claims / nullif(d.d_benes, 0)                           AS d_claims_per_bene,
    d.d_brand_claims / nullif(d.d_claims, 0)                    AS d_brand_share,
    d.d_opioid_claims / nullif(d.d_claims, 0)                   AS d_opioid_share,
    d.d_opioid_la_claims / nullif(d.d_opioid_claims, 0)         AS d_opioid_la_share,
    d.d_antibiotic_claims / nullif(d.d_claims, 0)               AS d_antibiotic_share,
    d.d_antipsychotic_claims / nullif(d.d_claims, 0)            AS d_antipsychotic_share,
    d.d_lis_claims / nullif(d.d_claims, 0)                      AS d_lis_share
FROM partb b
LEFT JOIN partd d USING (year, npi);
