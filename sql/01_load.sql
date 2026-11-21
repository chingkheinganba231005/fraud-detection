-- Load the raw files. Every column is read as text so that suppressed cells (CMS blanks
-- counts under 11) and stray values never stop the load, then the columns we use are cast.

CREATE OR REPLACE TABLE partb AS
SELECT
    CAST(regexp_extract(filename, 'partb_([0-9]{4})', 1) AS INTEGER) AS year,
    CAST(Rndrng_NPI AS BIGINT)                    AS npi,
    Rndrng_Prvdr_Type                             AS provider_type,
    Rndrng_Prvdr_Ent_Cd                           AS entity,
    Rndrng_Prvdr_State_Abrvtn                     AS state,
    TRY_CAST(Rndrng_Prvdr_RUCA AS DOUBLE)         AS ruca,
    Rndrng_Prvdr_Mdcr_Prtcptg_Ind                 AS participating,
    TRY_CAST(Tot_HCPCS_Cds AS DOUBLE)             AS b_codes,
    TRY_CAST(Tot_Benes AS DOUBLE)                 AS b_benes,
    TRY_CAST(Tot_Srvcs AS DOUBLE)                 AS b_services,
    TRY_CAST(Tot_Sbmtd_Chrg AS DOUBLE)            AS b_charges,
    TRY_CAST(Tot_Mdcr_Alowd_Amt AS DOUBLE)        AS b_allowed,
    TRY_CAST(Tot_Mdcr_Pymt_Amt AS DOUBLE)         AS b_payment,
    TRY_CAST(Drug_Mdcr_Pymt_Amt AS DOUBLE)        AS b_drug_payment,
    TRY_CAST(Bene_Avg_Age AS DOUBLE)              AS bene_age,
    TRY_CAST(Bene_Age_LT_65_Cnt AS DOUBLE)        AS bene_lt65,
    TRY_CAST(Bene_Age_GT_84_Cnt AS DOUBLE)        AS bene_gt84,
    TRY_CAST(Bene_Feml_Cnt AS DOUBLE)             AS bene_female,
    TRY_CAST(Bene_Dual_Cnt AS DOUBLE)             AS bene_dual,
    TRY_CAST(Bene_Avg_Risk_Scre AS DOUBLE)        AS risk_score
FROM read_csv('{raw}/partb_*.csv', all_varchar = true, filename = true, union_by_name = true)
WHERE TRY_CAST(Rndrng_NPI AS BIGINT) IS NOT NULL;

CREATE OR REPLACE TABLE partd AS
SELECT
    CAST(regexp_extract(filename, 'partd_([0-9]{4})', 1) AS INTEGER) AS year,
    CAST(Prscrbr_NPI AS BIGINT)                   AS npi,
    TRY_CAST(Tot_Clms AS DOUBLE)                  AS d_claims,
    TRY_CAST(Tot_Drug_Cst AS DOUBLE)              AS d_cost,
    TRY_CAST(Tot_Day_Suply AS DOUBLE)             AS d_days,
    TRY_CAST(Tot_Benes AS DOUBLE)                 AS d_benes,
    TRY_CAST(Brnd_Tot_Clms AS DOUBLE)             AS d_brand_claims,
    TRY_CAST(Opioid_Tot_Clms AS DOUBLE)           AS d_opioid_claims,
    TRY_CAST(Opioid_LA_Tot_Clms AS DOUBLE)        AS d_opioid_la_claims,
    TRY_CAST(Antbtc_Tot_Clms AS DOUBLE)           AS d_antibiotic_claims,
    TRY_CAST(Antpsyct_GE65_Tot_Clms AS DOUBLE)    AS d_antipsychotic_claims,
    TRY_CAST(LIS_Tot_Clms AS DOUBLE)              AS d_lis_claims
FROM read_csv('{raw}/partd_*.csv', all_varchar = true, filename = true, union_by_name = true)
WHERE TRY_CAST(Prscrbr_NPI AS BIGINT) IS NOT NULL;

-- One row per exclusion with an NPI. Most old exclusions have no NPI (it is 0 in the file)
-- and cannot be linked to claims, which is a limitation of the public list.
CREATE OR REPLACE TABLE leie AS
SELECT
    CAST(NPI AS BIGINT)                           AS npi,
    EXCLTYPE                                      AS excltype,
    strptime(EXCLDATE, '%Y%m%d')::DATE            AS excldate,
    GENERAL                                       AS general_category,
    SPECIALTY                                     AS leie_specialty
FROM read_csv('{raw}/leie_updated.csv', all_varchar = true)
WHERE TRY_CAST(NPI AS BIGINT) > 0;
