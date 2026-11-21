import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

DATA = Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def sample_db(tmp_path_factory):
    """The full SQL pipeline run on the committed sample of real CMS and OIG rows."""
    from fraud import config, db

    old = config.RAW
    config.RAW = DATA
    con = db.connect(tmp_path_factory.mktemp("db") / "sample.duckdb")
    for name in ("01_load.sql", "02_labels.sql", "03_provider_year.sql", "04_peer_stats.sql",
                 "05_peer_z.sql"):
        db.run(con, name)
    config.RAW = old
    yield con
    con.close()
