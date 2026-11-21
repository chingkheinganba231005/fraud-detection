-- Median and interquartile range of each metric among providers of the same type in the
-- same year, plus an all-provider row used when a peer group is small. No labels are used,
-- so computing these on the test year does not leak anything.
-- One wide row per (year, provider type): columns n__<metric>, med__<metric>, iqr__<metric>.

CREATE OR REPLACE TABLE peer_stats AS
SELECT year, provider_type, {peer_aggregates}
FROM provider_year
GROUP BY year, provider_type
UNION ALL
SELECT year, '__all__' AS provider_type, {peer_aggregates}
FROM provider_year
GROUP BY year;
