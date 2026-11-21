-- Robust z-score of each metric against the provider's peer group:
--   (value - peer median) / peer IQR, clipped to [-50, 50].
-- The scale is the peer IQR, or the peer standard deviation when the IQR is zero. Peer
-- groups with fewer than {min_peers} providers, or no spread at all, fall back to all providers.
-- A missing value or a missing scale gives a missing z-score, never the clip value.

CREATE OR REPLACE TABLE peer_z AS
SELECT f.year, f.npi, {z_expressions}
FROM provider_year f
LEFT JOIN peer_stats p ON p.year = f.year AND p.provider_type = f.provider_type
LEFT JOIN peer_stats a ON a.year = f.year AND a.provider_type = '__all__';
