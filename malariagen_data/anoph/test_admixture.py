"""
Tests for AnophelesAdmixtureAnalysis.

Structure
---------
Unit tests (``test_nmf_*``)
    Test the :func:`novel_run_admixture_nmf` core function using synthetic NumPy
    data only.  These run without GCS credentials and without the simulated
    fixture data; they work in any environment that has ``malariagen_data``
    installed (so the module can be imported).

Integration test (``test_admixture_*``)
    Uses the simulated fixture infrastructure defined in
    ``tests/anoph/conftest.py``.  To run these, either:

    * Moving/copying this file to ``tests/anoph/test_admixture.py`` and run
      ``poetry run pytest tests/anoph/test_admixture.py`` from the
      repository root, **or**
    * Run directly from ``technical_test/part1/`` after manually injecting
      the fixture via ``conftest.py`` (please see README for details).

Running the unit tests only
---------------------------
From the repository root::

    poetry run pytest technical_test/part1/test_admixture.py -v -k "nmf"
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Ensure the part1 directory is on the Python path so that admixture.py
# can be imported both when running pytest from the repo root and when
# running it from within technical_test/part1/.
# ---------------------------------------------------------------------------
_PART1_DIR = str(Path(__file__).parent)
if _PART1_DIR not in sys.path:
    sys.path.insert(0, _PART1_DIR)

from admixture import AnophelesAdmixtureAnalysis, novel_run_admixture_nmf  # noqa: E402


# ===========================================================================
# Unit tests — novel_run_admixture_nmf
# These tests are fully self-contained; no fixtures or GCS access required.
# ===========================================================================

# Class-based tests are used here to group the NMF tests together, but pytest
# will still discover and run them without needing to instantiate the class.


class TestRunAdmixtureNmf:
    """Unit tests for the pure-NumPy NMF core function."""

    def test_output_shapes(self):
        """Q must be (n_samples, k) and H must be (k, n_variants)."""
        rng = np.random.default_rng(0)
        X = rng.uniform(0, 2, (40, 500)).astype(np.float32)
        k = 3
        Q, H = novel_run_admixture_nmf(X, k=k, max_iter=50, random_seed=0)

        assert Q.shape == (40, k), f"Expected Q shape (40, {k}), got {Q.shape}"
        assert H.shape == (k, 500), f"Expected H shape ({k}, 500), got {H.shape}"

    def test_ancestry_proportions_sum_to_one(self):
        """Each sample's ancestry proportions must sum to 1 (up to float tolerance)."""
        rng = np.random.default_rng(42)
        X = rng.uniform(0, 2, (30, 200)).astype(np.float32)
        Q, _ = novel_run_admixture_nmf(X, k=2, max_iter=100, random_seed=42)

        np.testing.assert_allclose(
            Q.sum(axis=1),
            1.0,
            atol=1e-5,
            err_msg="Ancestry proportions do not sum to 1 for every sample",
        )

    def test_proportions_non_negative(self):
        """All ancestry proportions must be ≥ 0."""
        rng = np.random.default_rng(7)
        X = rng.uniform(0, 2, (50, 300)).astype(np.float32)
        Q, _ = novel_run_admixture_nmf(X, k=4, max_iter=50, random_seed=7)

        assert np.all(
            Q >= -1e-10
        ), f"Found {np.sum(Q < 0)} negative ancestry proportions"

    def test_recovers_two_well_separated_populations(self):
        """
        With two clearly differentiated populations, NMF should assign
        dominant ancestry to opposite populations for each group.
        """
        rng = np.random.default_rng(123)
        n, m = 60, 400

        # Pop1: low minor-allele frequency; Pop2: high minor-allele frequency.
        p1 = rng.uniform(0.05, 0.20, m)
        p2 = rng.uniform(0.80, 0.95, m)

        X = np.zeros((n, m), dtype=np.float32)
        half = n // 2
        for i in range(half):
            X[i] = rng.binomial(2, p1)  # pop1 samples
        for i in range(half, n):
            X[i] = rng.binomial(2, p2)  # pop2 samples

        Q, _ = novel_run_admixture_nmf(X, k=2, max_iter=300, random_seed=123)

        # The dominant ancestry column should differ between the two groups.
        dominant_pop1 = np.argmax(Q[:half].mean(axis=0))
        dominant_pop2 = np.argmax(Q[half:].mean(axis=0))

        assert dominant_pop1 != dominant_pop2, (
            "NMF did not separate the two populations: "
            f"pop1 dominant col={dominant_pop1}, pop2 dominant col={dominant_pop2}"
        )

    def test_reproducibility(self):
        """Same random_seed must always produce identical Q."""
        rng = np.random.default_rng(99)
        X = rng.uniform(0, 2, (25, 150)).astype(np.float32)

        Q1, _ = novel_run_admixture_nmf(X, k=3, max_iter=80, random_seed=5)
        Q2, _ = novel_run_admixture_nmf(X, k=3, max_iter=80, random_seed=5)

        np.testing.assert_array_equal(Q1, Q2, err_msg="Results not reproducible")

    def test_different_seeds_differ(self):
        """Different random seeds should generally produce different Q."""
        rng = np.random.default_rng(0)
        X = rng.uniform(0, 2, (20, 100)).astype(np.float32)

        Q1, _ = novel_run_admixture_nmf(X, k=2, max_iter=50, random_seed=1)
        Q2, _ = novel_run_admixture_nmf(X, k=2, max_iter=50, random_seed=999)

        # Very unlikely to be exactly equal with different seeds.
        assert not np.allclose(
            Q1, Q2
        ), "Different seeds produced identical results (unexpected)"

    def test_k_equals_1_returns_all_ones(self):
        """With K=1, every sample should have 100% ancestry in pop1."""
        rng = np.random.default_rng(0)
        X = rng.uniform(0.1, 1.9, (10, 50)).astype(np.float32)
        Q, _ = novel_run_admixture_nmf(X, k=1, max_iter=50, random_seed=0)

        assert Q.shape == (10, 1)
        np.testing.assert_allclose(
            Q.flatten(),
            1.0,
            atol=1e-6,
            err_msg="With K=1, all proportions should be 1.0",
        )


# ===========================================================================
# Integration tests — AnophelesAdmixtureAnalysis.admixture()
#
# These require the simulated Ag3 fixture from tests/anoph/conftest.py.
# They are skipped automatically when that fixture is not available
# (e.g., when running this file from technical_test/part1/ directly).
# ===========================================================================

try:
    from malariagen_data import ag3 as _ag3

    _HAS_AG3 = True
except ImportError:
    _HAS_AG3 = False


def _make_admixture_api(fixture):
    """Create a minimal AnophelesAdmixtureAnalysis backed by simulated data."""
    return AnophelesAdmixtureAnalysis(
        url=fixture.url,
        public_url=fixture.url,
        config_path=_ag3.CONFIG_PATH,
        major_version_number=_ag3.MAJOR_VERSION_NUMBER,
        major_version_path=_ag3.MAJOR_VERSION_PATH,
        pre=True,
        aim_metadata_dtype={
            "aim_species_fraction_arab": "float64",
            "aim_species_fraction_colu": "float64",
            "aim_species_fraction_colu_no2l": "float64",
            "aim_species_gambcolu_arabiensis": object,
            "aim_species_gambiae_coluzzii": object,
            "aim_species": object,
        },
        gff_gene_type="gene",
        gff_gene_name_attribute="Name",
        gff_default_attributes=("ID", "Parent", "Name", "description"),
        default_site_mask="gamb_colu_arab",
        results_cache=fixture.results_cache_path.as_posix(),
        taxon_colors=_ag3.TAXON_COLORS,
        virtual_contigs=_ag3.VIRTUAL_CONTIGS,
    )


# The integration fixtures are only available when running inside tests/anoph/.
# We use pytest.fixture with indirect=True pattern — if `ag3_sim_fixture` is
# not discoverable, the test below will be collected but skipped via the
# `pytest.importorskip`-style guard in the fixture function body.


@pytest.fixture
def ag3_admixture_api(ag3_sim_fixture):
    """
    Fixture: AnophelesAdmixtureAnalysis backed by simulated Ag3 data.

    Requires ``ag3_sim_fixture`` from tests/anoph/conftest.py.
    Place this file in tests/anoph/ to activate this fixture.
    """
    return _make_admixture_api(ag3_sim_fixture)


def test_admixture_returns_dataframe(ag3_admixture_api):
    """
    Integration: admixture() returns a well-formed DataFrame.

    NOTE: This test requires the ag3_sim_fixture defined in
    tests/anoph/conftest.py.  Place this file in tests/anoph/ to run it.
    """
    api = ag3_admixture_api
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    contig = api.contigs[0]
    k = 2

    df = api.admixture(
        region=contig,
        k=k,
        sample_sets=all_sample_sets[:2],
        n_snps=200,
        min_minor_ac=1,
        max_missing_an=None,
        random_seed=0,
        max_iter=50,
    )

    # Output type.
    assert isinstance(df, pd.DataFrame), "admixture() should return a DataFrame"

    # Required columns present.
    assert "sample_id" in df.columns
    for i in range(1, k + 1):
        assert f"pop{i}" in df.columns, f"Missing column pop{i}"

    # Non-empty.
    assert len(df) > 0, "DataFrame should have at least one row"

    # Ancestry proportions sum to 1 per sample.
    q_cols = [f"pop{i}" for i in range(1, k + 1)]
    row_sums = df[q_cols].sum(axis=1)
    np.testing.assert_allclose(
        row_sums.values,
        1.0,
        atol=1e-4,
        err_msg="Ancestry proportions do not sum to 1",
    )

    # All proportions non-negative.
    assert (df[q_cols].values >= -1e-6).all(), "Negative ancestry proportions found"
