-- Label every provider-year with what the OIG did in the next {window} calendar years.
--   label = 1   excluded for a fraud-related reason inside the window
--   dropped     already excluded before or during the billing year (should not be billing),
--               or excluded only for a non-fraud reason inside the window (ambiguous)
-- Everyone else is a negative, which really means "not caught": undetected fraud is in there.

CREATE OR REPLACE TABLE labels AS
WITH provider_years AS (
    SELECT DISTINCT year, npi FROM partb
),
hits AS (
    SELECT
        p.year,
        p.npi,
        max(CASE WHEN l.excldate <= make_date(p.year, 12, 31) THEN 1 ELSE 0 END) AS excluded_before,
        max(CASE WHEN l.excltype IN ({fraud_types})
                  AND l.excldate BETWEEN make_date(p.year + 1, 1, 1) AND make_date(p.year + {window}, 12, 31)
                 THEN 1 ELSE 0 END) AS fraud_in_window,
        max(CASE WHEN l.excltype NOT IN ({fraud_types})
                  AND l.excldate BETWEEN make_date(p.year + 1, 1, 1) AND make_date(p.year + {window}, 12, 31)
                 THEN 1 ELSE 0 END) AS other_in_window,
        min(CASE WHEN l.excltype IN ({fraud_types})
                  AND l.excldate BETWEEN make_date(p.year + 1, 1, 1) AND make_date(p.year + {window}, 12, 31)
                 THEN l.excldate END) AS exclusion_date,
        arg_min(CASE WHEN l.excltype IN ({fraud_types})
                      AND l.excldate BETWEEN make_date(p.year + 1, 1, 1) AND make_date(p.year + {window}, 12, 31)
                     THEN l.excltype END, l.excldate) AS exclusion_type
    FROM provider_years p
    LEFT JOIN leie l USING (npi)
    GROUP BY p.year, p.npi
)
SELECT
    year,
    npi,
    fraud_in_window AS label,
    CASE
        WHEN excluded_before = 1 THEN 'excluded before or during the billing year'
        WHEN fraud_in_window = 0 AND other_in_window = 1 THEN 'excluded for a non-fraud reason'
    END AS drop_reason,
    exclusion_date,
    exclusion_type
FROM hits;
