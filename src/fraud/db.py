"""Run the SQL files in sql/ against a DuckDB connection."""

import duckdb

from . import config


def sql_text(name: str) -> str:
    """Read sql/<name> and fill the placeholders shared by the queries."""
    aggregates = ",\n    ".join(
        f"count({m}) AS n__{m}, median({m}) AS med__{m}, "
        f"quantile_cont({m}, 0.75) - quantile_cont({m}, 0.25) AS iqr__{m}, "
        f"stddev_samp({m}) AS sd__{m}"
        for m in config.PEER_METRICS)

    def scale(t, m):
        # The IQR, or the standard deviation when more than half the group shares one value
        # (for example no office-administered drugs at all) and the IQR is zero.
        return f"coalesce(nullif({t}.iqr__{m}, 0), nullif({t}.sd__{m}, 0))"

    def z(m):
        return (f"CASE WHEN p.n__{m} >= {config.MIN_PEERS} AND {scale('p', m)} IS NOT NULL "
                f"THEN (f.{m} - p.med__{m}) / {scale('p', m)} "
                f"ELSE (f.{m} - a.med__{m}) / {scale('a', m)} END")

    # greatest() and least() skip NULLs in DuckDB, so a missing z-score has to be kept
    # missing explicitly or it would come out as the clip value.
    z_expressions = ",\n    ".join(
        f"CASE WHEN ({z(m)}) IS NULL THEN NULL ELSE greatest(-50, least(50, {z(m)})) END AS z_{m}"
        for m in config.PEER_METRICS)
    fraud = ", ".join(f"'{t}'" for t in config.FRAUD_EXCLUSION_TYPES)
    text = (config.SQL / name).read_text()
    return (text.replace("{raw}", str(config.RAW))
                .replace("{peer_aggregates}", aggregates)
                .replace("{z_expressions}", z_expressions)
                .replace("{fraud_types}", fraud)
                .replace("{window}", str(config.LABEL_WINDOW_YEARS))
                .replace("{min_peers}", str(config.MIN_PEERS)))


def run(con: duckdb.DuckDBPyConnection, name: str) -> None:
    for statement in sql_text(name).split(";"):
        if statement.strip():
            con.execute(statement).fetchall()


def connect(path=None) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(str(path or config.DB_PATH))
    # Row order never matters here; dropping it lets DuckDB stream instead of spilling to disk.
    con.execute("SET preserve_insertion_order = false")
    return con


def features_query(where: str = "TRUE") -> str:
    cols = ", ".join(f"f.{c}" for c in config.RAW_FEATURES + config.CATEGORICAL_FEATURES)
    zcols = ", ".join(f"z.{c}" for c in config.Z_FEATURES)
    return f"""
        SELECT f.year, f.npi, f.state, f.b_payment, f.d_cost, {cols}, {zcols}
        FROM provider_year f JOIN peer_z z USING (year, npi)
        WHERE {where}
    """


def build_features(con: duckdb.DuckDBPyConnection) -> None:
    """provider_year and peer_z from the partb/partd tables already in the connection."""
    run(con, "03_provider_year.sql")
    run(con, "05_peer_z.sql")


def load_split(con: duckdb.DuckDBPyConnection, years) -> "pd.DataFrame":
    """Features and labels for the given years, without the dropped provider-years."""
    years = tuple(years) if len(years) > 1 else f"({years[0]})"
    return con.execute(f"""
        SELECT x.*, l.label, l.exclusion_type, l.exclusion_date
        FROM ({features_query(f"f.year IN {years}")}) x
        JOIN labels l USING (year, npi)
        WHERE l.drop_reason IS NULL
    """).df()


def features_from_raw(partb: "pd.DataFrame", partd: "pd.DataFrame",
                      peer_stats: "pd.DataFrame") -> "pd.DataFrame":
    """Run the feature SQL on rows that are not in the main database (synthetic copies,
    or a provider sent to the scoring service), against given peer statistics."""
    con = duckdb.connect()
    con.register("b_in", partb)
    con.register("d_in", partd)
    con.register("s_in", peer_stats)
    con.execute("CREATE TABLE partb AS SELECT * FROM b_in")
    con.execute("CREATE TABLE partd AS SELECT * FROM d_in")
    con.execute("CREATE TABLE peer_stats AS SELECT * FROM s_in")
    build_features(con)
    return con.execute(features_query()).df()
