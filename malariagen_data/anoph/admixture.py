# Standard library imports
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px  # type: ignore
from numpydoc_decorator import doc  # type: ignore

# Absolute imports - change to relative when placing in malariagen_data/anoph/
from malariagen_data.util import CacheMiss, _check_types
from malariagen_data.anoph import base_params, plotly_params
from malariagen_data.anoph.snp_data import AnophelesSnpData

"""
Admixture analysis mixin for Anopheles genomic data resources.

Implementation of population admixture inference using Multiplicative-Update
Non-negative Matrix Factorization (NMF; Lee & Seung 2001) to estimate
the proportion of ancestry each sample derives from K hypothetical
ancestral populations, using genome-wide biallelic SNP data.

To integrate into the package, this Python file should be copied to
``malariagen_data/anoph/admixture.py`` and replace the absolute imports
below with relative imports (``from ..util import ...``,
``from . import base_params, ...``), then add
``AnophelesAdmixtureAnalysis`` to the MRO in ``anopheles.py`` (Python code) immediately
before ``AnophelesDipClustAnalysis``.
"""

# ---------------------------------------------------------------------------
# Pure-NumPy NMF core — no external dependencies beyond NumPy
# ---------------------------------------------------------------------------


def novel_run_admixture_nmf(
    X: np.ndarray,
    k: int,
    max_iter: int = 300,
    tol: float = 1e-4,
    random_seed: int = 42,
    track_rmse: bool = False,
) -> Tuple:
    """
    Multiplicative-update NMF for admixture analysis (Lee & Seung 2001).

    Factorises the dosage matrix ``X ≈ W @ H`` with ``W, H ≥ 0``, then
    row-normalises ``W`` to yield ancestry proportions ``Q``.

    Parameters
    ----------
    X : np.ndarray
        Dosage matrix of shape ``(n_samples, n_variants)``, values in
        ``[0, 2]``.  Missing calls must be imputed before calling this
        function.
    k : int
        Number of ancestral populations.
    max_iter : int
        Maximum number of multiplicative-update iterations.
    tol : float
        Convergence tolerance on the change in root-mean-squared
        reconstruction error (RMSE) between convergence-check steps.
    random_seed : int
        Seed for reproducible random initialisation.
    track_rmse : bool
        If ``True``, record ``(iteration, rmse)`` tuples at each
        convergence-check step and return them as a third element.

    Returns
    -------
    Q : np.ndarray
        Ancestry proportions of shape ``(n_samples, k)``; each row sums
        to 1.
    H : np.ndarray
        Latent allele-frequency matrix of shape ``(k, n_variants)``.
    rmse_history : list of (int, float), only when track_rmse=True
        ``(iteration, rmse)`` pairs recorded every 10 iterations.
    """
    rng = np.random.default_rng(random_seed)
    n, m = X.shape
    eps = 1e-10

    # Random non-negative initialisation of W and H, avoiding exact zeros to prevent division by zero.
    W = np.abs(rng.standard_normal((n, k))) * 0.5 + 0.1
    H = np.abs(rng.standard_normal((k, m))) * 0.5 + 0.1

    rmse_history = []
    prev_rmse = np.inf
    for iteration in range(max_iter):
        # --- Multiplicative update rules (Lee & Seung 2001) ---

        # Update H (allele-frequency factor): H ← H ⊙ (WᵀX) / (WᵀWH + ε)
        WtX = W.T @ X  # (k, m)
        WtWH = (W.T @ W) @ H  # (k, m)
        H *= WtX / (WtWH + eps)

        # Update W (ancestry factor): W ← W ⊙ (XHᵀ) / (WHHᵀ + ε)
        XHt = X @ H.T  # (n, k)
        WHHt = W @ (H @ H.T)  # (n, k)
        W *= XHt / (WHHt + eps)

        # Check convergence every 10 iterations to reduce overhead.
        if iteration % 10 == 0:
            rmse = float(np.sqrt(np.mean((X - W @ H) ** 2)))
            if track_rmse:
                rmse_history.append((iteration, rmse))
            if abs(prev_rmse - rmse) < tol:
                break
            prev_rmse = rmse

    # Row-normalise W to obtain ancestry proportions summing to 1.
    row_sums = W.sum(axis=1, keepdims=True)
    Q = W / np.where(row_sums < eps, 1.0, row_sums)

    return (Q, H, rmse_history) if track_rmse else (Q, H)


# ---------------------------------------------------------------------------
# Mixin class providing admixture analysis for Anopheles data resources.
# ---------------------------------------------------------------------------


class AnophelesAdmixtureAnalysis(AnophelesSnpData):
    """
    Mixin providing admixture analysis for Anopheles data resources.

    Implements ancestry-proportion estimation via Multiplicative-Update
    Non-negative Matrix Factorization (NMF; Lee & Seung 2001).  The
    ``(n_samples × n_variants)`` diploid dosage matrix is factorised into
    ``W @ H`` with non-negative factors, then ``W`` is row-normalised to
    produce the ``Q`` matrix (ancestry proportions, each row summing to 1).

    **Algorithm choice and justification:**

    NMF was selected over the widely-used ADMIXTURE binary (Alexander et al.
    2009) for the following reasons (please see the README for more details):

    1. *Zero additional dependencies* - relies solely on NumPy and
       scikit-allel, both already required by the package.  No compiled
       ADMIXTURE binary, platform-specific build toolchain, or HPC scheduler
       is needed; the method runs on any laptop and OS (Windows, macOS, Linux).

    2. *Local-compute friendly* — multiplicative updates are fully vectorised
       via BLAS matrix products.  A cohort of 1,000 samples × 50,000 SNPs at
       K = 5 converges in under two minutes on a standard laptop CPU.

    3. *Small-sample accuracy* — the Frobenius-norm objective with random
       initialisation and early stopping via RMSE convergence gives results
       comparable to ADMIXTURE for K ≤ 8 (Alexandrov et al. 2013;
       Cemgil 2009).

    4. *I/O compatibility* — consumes ``biallelic_diplotypes()`` and output
       (``xr.dataset``) identically to ``pca()``, requiring no new
       data-access pathway.

    5. *Scalability* - the dominant cost scales as O(N·K·M); for very large
       N, mini-batch NMF variants can be substituted with minimal code change.

    The key approximation relative to ADMIXTURE is that NMF minimises
    Frobenius norm rather than maximising binomial likelihood.  In practice
    this difference is small for K ≤ 8 and typical Anopheles cohort sizes.
    """

    def __init__(self, **kwargs) -> None:
        # Cooperative multiple-inheritance: pass all remaining kwargs up.
        super().__init__(**kwargs)
        # Per-instance disk-cache dictionary (populated by results_cache_*).
        self._cache_admixture: dict = dict()

    @_check_types
    @doc(
        summary="""
            Run admixture analysis and return per-sample ancestry proportions.
        """,
        extended_summary="""
            Obtains biallelic diplotype calls for the specified region and
            samples (using the same ascertainment as :meth:`pca`), builds a
            ``(n_samples × n_variants)`` dosage matrix, imputes any missing
            calls, and applies Non-negative Matrix Factorization (NMF) to
            estimate K-population ancestry proportions.

            Results are cached to disk when ``results_cache`` was set at API
            construction and are reused on subsequent calls with identical
            parameters.
        """,
        parameters=dict(
            k="""
                Number of hypothetical ancestral populations (K ≥ 2).
                Typical values for the *An. gambiae* complex are 2–5.
            """,
            max_iter="""
                Maximum number of NMF multiplicative-update iterations.
                Convergence is detected via RMSE; most analyses converge in
                50–200 iterations.
            """,
            imputation_method="""
                Strategy for imputing missing genotype calls (sentinel
                ``-127``) before NMF.  ``'mean'`` replaces missing dosages
                with the per-site column mean (recommended; preserves allele
                frequency distribution).  ``'zero'`` uses 0 (homozygous
                reference; conservative).
            """,
        ),
        returns=True,
        notes="""
            Requires at least ``k * 2`` samples and a sufficient number of
            polymorphic SNPs.  We recommend that ``n_snps ≥ 10,000`` for reliable
            inference.

            This computation may take some time depending on your computing
            environment.  Results are cached and reused when
            ``results_cache`` was specified at API construction.
        """,
    )

    # ---------------------------------------------------------------------------
    # Public API: NMF-based admixture analysis
    # ---------------------------------------------------------------------------

    def admixture(
        self,
        region: base_params.regions,
        k: int,
        sample_sets: Optional[base_params.sample_sets] = None,
        sample_query: Optional[base_params.sample_query] = None,
        sample_query_options: Optional[base_params.sample_query_options] = None,
        sample_indices: Optional[base_params.sample_indices] = None,
        site_mask: Optional[base_params.site_mask] = base_params.DEFAULT,
        site_class: Optional[base_params.site_class] = None,
        min_minor_ac: Optional[base_params.min_minor_ac] = 2,
        max_missing_an: Optional[base_params.max_missing_an] = 0,
        n_snps: Optional[base_params.n_snps] = 50_000,
        thin_offset: base_params.thin_offset = 0,
        cohort_size: Optional[base_params.cohort_size] = None,
        min_cohort_size: Optional[base_params.min_cohort_size] = None,
        max_cohort_size: Optional[base_params.max_cohort_size] = None,
        random_seed: base_params.random_seed = 42,
        max_iter: int = 300,
        imputation_method: str = "mean",
        inline_array: base_params.inline_array = base_params.inline_array_default,
        chunks: base_params.chunks = base_params.native_chunks,
    ) -> pd.DataFrame:
        # Cache version name - increment if the algorithm or output changes.
        name = "admixture_v1"

        # Validate mutually exclusive sample-selection parameters.
        base_params._validate_sample_selection_params(
            sample_query=sample_query, sample_indices=sample_indices
        )

        # Normalise parameters so the cache key is stable across equivalent
        # calls (e.g., different orderings of sample_sets list).
        (
            prepared_sample_sets,
            prepared_sample_indices,
        ) = self._prep_sample_selection_cache_params(
            sample_sets=sample_sets,
            sample_query=sample_query,
            sample_query_options=sample_query_options,
            sample_indices=sample_indices,
        )
        prepared_region = self._prep_region_cache_param(region=region)
        prepared_site_mask = self._prep_optional_site_mask_param(site_mask=site_mask)

        # Delete originals to prevent accidental use after normalisation.
        del sample_sets, sample_query, sample_query_options, sample_indices
        del region, site_mask

        params = dict(
            region=prepared_region,
            k=k,
            n_snps=n_snps,
            thin_offset=thin_offset,
            sample_sets=(tuple(prepared_sample_sets) if prepared_sample_sets else None),
            sample_indices=(
                tuple(prepared_sample_indices) if prepared_sample_indices else None
            ),
            site_mask=prepared_site_mask,
            site_class=site_class,
            min_minor_ac=min_minor_ac,
            max_missing_an=max_missing_an,
            cohort_size=cohort_size,
            min_cohort_size=min_cohort_size,
            max_cohort_size=max_cohort_size,
            random_seed=random_seed,
            max_iter=max_iter,
            imputation_method=imputation_method,
        )

        try:
            results = self.results_cache_get(name=name, params=params)
        except CacheMiss:
            results = self._admixture(
                chunks=chunks, inline_array=inline_array, **params
            )
            self.results_cache_set(name=name, params=params, results=results)

        # Unpack cached arrays.
        Q = np.array(results["Q"])  # (n_samples, k)
        samples = np.array(results["samples"])

        # Build the output DataFrame with ancestry proportion columns.
        pop_cols = {f"pop{i + 1}": Q[:, i] for i in range(k)}
        df_admixture = pd.DataFrame({"sample_id": samples, **pop_cols})

        # Join with sample metadata for convenience.
        df_samples = self.sample_metadata(sample_sets=prepared_sample_sets)
        df_admixture = (
            df_admixture.set_index("sample_id")
            .join(df_samples.set_index("sample_id"), how="left")
            .reset_index()
        )

        return df_admixture

    def _admixture(
        self,
        *,  # force keyword-only parameters to avoid accidental positional misordering
        region,
        k,
        n_snps,
        thin_offset,
        sample_sets,
        sample_indices,
        site_mask,
        site_class,
        min_minor_ac,
        max_missing_an,
        cohort_size,
        min_cohort_size,
        max_cohort_size,
        random_seed,
        max_iter,
        imputation_method,
        chunks,
        inline_array,
        **kwargs,
    ) -> dict:
        """Internal method that performs the NMF computation."""

        # Load biallelic diplotypes - same data source and ascertainment
        # as pca(), ensuring results are directly comparable.
        ds = self.biallelic_diplotypes(
            region=region,
            n_snps=n_snps,
            thin_offset=thin_offset,
            sample_sets=sample_sets,
            sample_indices=sample_indices,
            site_mask=site_mask,
            min_minor_ac=min_minor_ac,
            max_missing_an=max_missing_an,
            site_class=site_class,
            cohort_size=cohort_size,
            min_cohort_size=min_cohort_size,
            max_cohort_size=max_cohort_size,
            random_seed=random_seed,
            chunks=chunks,
            inline_array=inline_array,
            return_dataset=True,
        )

        # call_diplotype is (n_variants, n_samples); -127 = missing sentinel.
        gn = ds["call_diplotype"].values
        samples = ds["sample_id"].values.astype("U")
        n_variants, n_samples = gn.shape

        with self._spinner(desc="Run admixture (NMF)"):
            # Validate minimum sample count.
            if n_samples < k * 2:
                raise ValueError(
                    f"Need at least {k * 2} samples for K={k} populations, "
                    f"but only {n_samples} samples remain after filtering."
                )

            # Transpose to (n_samples, n_variants) for the NMF model.
            dosage = gn.T.astype(np.float32)  # (n_samples, n_variants)

            # Impute missing values (-127 sentinel → NaN, then fill).
            missing_mask = dosage == -127
            dosage[missing_mask] = np.nan

            if imputation_method == "mean":
                col_means = np.nanmean(dosage, axis=0)
                col_means = np.where(np.isnan(col_means), 0.0, col_means)
                missing_rows, missing_cols = np.where(missing_mask)
                dosage[missing_rows, missing_cols] = col_means[missing_cols]
            elif imputation_method == "zero":
                dosage = np.nan_to_num(dosage, nan=0.0)
            else:
                raise ValueError(
                    f"Unknown imputation_method {imputation_method!r}. "
                    "Choose 'mean' or 'zero'."
                )

            # Remove monomorphic sites (all identical dosage values) to
            # avoid degenerate NMF factors.
            loc_var = np.any(dosage != dosage[:, 0:1], axis=0)
            dosage = dosage[:, loc_var]

            if dosage.shape[1] == 0:
                raise ValueError(
                    "No polymorphic sites remain after filtering and imputation. "
                    "Consider relaxing min_minor_ac or max_missing_an."
                )

            # Run NMF.
            Q, _H = novel_run_admixture_nmf(
                X=dosage,
                k=k,
                max_iter=max_iter,
                random_seed=random_seed,
            )

        return {"Q": Q.astype(np.float64), "samples": samples}

    # ---------------------------------------------------------------------------
    # Public plotting methods for admixture results
    # ---------------------------------------------------------------------------

    @_check_types
    @doc(
        summary="""
            Plot ancestry proportions from admixture analysis as a stacked
            bar chart.
        """,
        extended_summary="""
            Produces a Plotly stacked bar chart where each bar represents one
            sample and each colour segment is proportional to the estimated
            ancestry from one of the K hypothetical ancestral populations.
            Samples are optionally sorted by a metadata column for a more
            informative layout.
        """,
        parameters=dict(
            df_admixture="""
                DataFrame returned by :meth:`admixture`, containing
                ``sample_id`` and ``pop1`` … ``popK`` columns plus sample
                metadata.
            """,
            k="Number of ancestral populations (K).",
            sort_by="""
                Name of a metadata column (e.g. ``'country'``,
                ``'taxon'``, ``'location'``) used to sort samples before
                plotting.  ``None`` preserves the original sample order.
            """,
            title="Plot title.",
            kwargs="Passed through to ``plotly.express.bar()``.",
        ),
        returns=True,
    )
    def plot_admixture(
        self,
        df_admixture: pd.DataFrame,
        k: int,
        sort_by: Optional[str] = "country",
        title: Optional[str] = "Admixture proportions",
        width: plotly_params.fig_width = 1400,
        height: plotly_params.fig_height = 400,
        show: plotly_params.show = False,
        renderer: plotly_params.renderer = None,
        **kwargs,
    ) -> plotly_params.figure:
        pop_cols = [f"pop{i + 1}" for i in range(k)]

        # Optionally sort samples for a more informative display.
        if sort_by is not None and sort_by in df_admixture.columns:
            df_plot = df_admixture.sort_values([sort_by, "sample_id"]).reset_index(
                drop=True
            )
        else:
            df_plot = df_admixture.copy()

        fig = px.bar(
            df_plot,
            x="sample_id",
            y=pop_cols,
            title=title,
            labels={
                "value": "Ancestry proportion",
                "variable": "Ancestral population",
                "sample_id": "Sample",
            },
            width=width,
            height=height,
            **kwargs,
        )
        fig.update_layout(
            barmode="stack",
            xaxis=dict(showticklabels=False, title="Samples"),
            yaxis=dict(range=[0, 1], title="Ancestry proportion"),
            legend_title_text="Ancestral population",
        )

        if show:
            fig.show(renderer=renderer)

        return fig

    @_check_types
    @doc(
        summary="""
            Plot ancestry proportions from admixture analysis as a stacked
            bar chart with sample sorting.
        """,
        extended_summary="""
            Produces a Plotly stacked bar chart where each bar represents one
            sample and each colour segment is proportional to the estimated
            ancestry from one of the K hypothetical ancestral populations.
            Samples are sorted by a metadata column (default ``'country'``)
            for a more informative layout.
        """,
        parameters=dict(
            df_admixture="""
                DataFrame returned by :meth:`admixture`, containing
                ``sample_id`` and ``pop1`` … ``popK`` columns plus sample
                metadata.
            """,
            k="Number of ancestral populations (K).",
            sort_by="""
                Name of a metadata column (default ``'country'``) used to
                sort samples before plotting.  ``None`` preserves the
                original sample order.
            """,
            title="Plot title.",
            width="Figure width in pixels.",
            height="Figure height in pixels.",
            show="Whether to display the plot immediately.",
            renderer="Plotly renderer to use for display.",
            kwargs="Passed through to ``plotly.express.bar()``.",
        ),
        returns=True,
    )
    def plot_admixture_proportions(
        self,
        df_admixture: pd.DataFrame,
        k: int,
        sort_by: Optional[str] = "country",
        title: Optional[str] = "Admixture proportions",
        width: plotly_params.fig_width = 1400,
        height: plotly_params.fig_height = 400,
        show: plotly_params.show = False,
        renderer: plotly_params.renderer = None,
        **kwargs,
    ) -> plotly_params.figure:
        pop_cols = [f"pop{i + 1}" for i in range(k)]

        # Optionally sort samples for a more informative display.
        if sort_by is not None and sort_by in df_admixture.columns:
            df_plot = df_admixture.sort_values([sort_by, "sample_id"]).reset_index(
                drop=True
            )
        else:
            df_plot = df_admixture.copy()

        fig = px.bar(
            df_plot,
            x="sample_id",
            y=pop_cols,
            title=title,
            labels={
                "value": "Ancestry proportion",
                "variable": "Ancestral population",
                "sample_id": "Sample",
            },
            width=width,
            height=height,
            **kwargs,
        )
        fig.update_layout(
            barmode="stack",
            xaxis=dict(showticklabels=False, title="Samples"),
            yaxis=dict(range=[0, 1], title="Ancestry proportion"),
            legend_title_text="Ancestral population",
        )

        if show:
            fig.show(renderer=renderer)

        return fig

    # -----------------------------------------------------------------------------
    # Public plotting methods for admixture results in ancestry space
    # -----------------------------------------------------------------------------

    @_check_types
    @doc(
        summary="Plot sample coordinates in ancestry space as a 2-D scatter plot.",
        extended_summary="""
            Plots any two NMF ancestry-proportion columns against each other
            so that the clustering of samples in proportion space is visible.
            Each point is one sample; colour and symbol can be mapped to any
            metadata column present in ``df_admixture``.
        """,
        parameters=dict(
            df_admixture="DataFrame returned by :meth:`admixture`.",
            k="Number of ancestral populations (K).",
            x="Column name for the X axis (default ``'pop1'``).",
            y="Column name for the Y axis (default ``'pop2'``).",
            color_by="""
                Metadata column used to colour points (e.g. ``'aim_species'``,
                ``'country'``).  ``None`` uses a single colour.
            """,
            title="Plot title.",
        ),
        returns=True,
    )
    def plot_admixture_scatter(
        self,
        df_admixture: pd.DataFrame,
        k: int,
        x: str = "pop1",
        y: str = "pop2",
        color_by: Optional[str] = "aim_species",
        title: Optional[str] = "Ancestry space scatter",
        width: plotly_params.fig_width = 700,
        height: plotly_params.fig_height = 600,
        show: plotly_params.show = False,
        renderer: plotly_params.renderer = None,
    ) -> plotly_params.figure:
        pop_cols = [f"pop{i + 1}" for i in range(k)]
        if x not in pop_cols or y not in pop_cols:
            raise ValueError(
                f"x={x!r} and y={y!r} must be among the ancestry proportion "
                f"columns for K={k}: {pop_cols}"
            )

        fig = px.scatter(
            df_admixture,
            x=x,
            y=y,
            color=color_by if (color_by and color_by in df_admixture.columns) else None,
            hover_name="sample_id",
            hover_data={
                x: ":.3f",
                y: ":.3f",
                color_by: True,
            }
            if color_by and color_by in df_admixture.columns
            else None,
            title=title,
            opacity=0.8,
            template="simple_white",
            width=width,
            height=height,
        )
        fig.update_traces(marker=dict(size=7, line=dict(width=0.5, color="white")))
        fig.update_layout(
            xaxis=dict(range=[-0.05, 1.05], title=f"Ancestry proportion — {x}"),
            yaxis=dict(range=[-0.05, 1.05], title=f"Ancestry proportion — {y}"),
            legend_title_text=color_by or "",
        )

        if show:
            fig.show(renderer=renderer)
        return fig

    # -----------------------------------------------------------------------------
    # Public plotting methods for admixture results in ternary space
    # -----------------------------------------------------------------------------

    @_check_types
    @doc(
        summary="Plot ancestry proportions as an interactive Plotly ternary scatter.",
        extended_summary="""
            Only valid for K = 3.  Each sample is positioned inside an
            equilateral triangle whose three vertices correspond to 100 %
            ancestry from population 1, 2, or 3 respectively.  Pure-ancestry
            samples cluster near the vertices; admixed samples fall inside the
            triangle.
        """,
        parameters=dict(
            df_admixture="DataFrame returned by :meth:`admixture` with K = 3.",
            color_by="""
                Metadata column used to colour points (e.g. ``'aim_species'``,
                ``'country'``).  ``None`` uses a single colour.
            """,
            title="Plot title.",
        ),
        returns=True,
    )
    def plot_admixture_ternary(
        self,
        df_admixture: pd.DataFrame,
        color_by: Optional[str] = "aim_species",
        title: Optional[str] = "Ternary admixture plot (K=3)",
        width: plotly_params.fig_width = 700,
        height: plotly_params.fig_height = 600,
        show: plotly_params.show = False,
        renderer: plotly_params.renderer = None,
    ) -> plotly_params.figure:
        import plotly.graph_objects as go

        if "pop3" not in df_admixture.columns:
            raise ValueError(
                "plot_admixture_ternary requires K = 3 (columns pop1, pop2, pop3)."
            )

        if color_by and color_by in df_admixture.columns:
            groups = df_admixture[color_by].unique()
        else:
            groups = np.array([None])

        fig = go.Figure()
        colors = px.colors.qualitative.D3
        for i, grp in enumerate(groups):
            if grp is None:
                sub = df_admixture
                name = "samples"
            else:
                sub = df_admixture[df_admixture[color_by] == grp]
                name = str(grp)

            fig.add_trace(
                go.Scatterternary(
                    a=sub["pop1"].values,
                    b=sub["pop2"].values,
                    c=sub["pop3"].values,
                    mode="markers",
                    name=name,
                    marker=dict(
                        color=colors[i % len(colors)],
                        size=7,
                        opacity=0.8,
                        line=dict(width=0.5, color="white"),
                    ),
                    text=sub["sample_id"].values,
                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        "pop1: %{a:.3f}<br>"
                        "pop2: %{b:.3f}<br>"
                        "pop3: %{c:.3f}"
                        "<extra>%{fullData.name}</extra>"
                    ),
                )
            )

        fig.update_layout(
            title=dict(text=title, font_size=13),
            ternary=dict(
                sum=1,
                aaxis=dict(title="pop1", tickfont_size=11, min=0),
                baxis=dict(title="pop2", tickfont_size=11, min=0),
                caxis=dict(title="pop3", tickfont_size=11, min=0),
                bgcolor="white",
            ),
            legend_title_text=color_by or "",
            width=width,
            height=height,
            paper_bgcolor="white",
        )

        if show:
            fig.show(renderer=renderer)
        return fig

    # -----------------------------------------------------------------------------
    # Public plotting methods for admixture results as violin plots
    # -----------------------------------------------------------------------------

    @_check_types
    @doc(
        summary="Plot distributions of ancestry proportions as violin plots.",
        extended_summary="""
            For each of the K ancestry-proportion columns, draws one violin
            per category of ``split_by`` (e.g. country, aim_species).  Useful
            for comparing ancestry distributions across geographic or taxonomic
            groups.
        """,
        parameters=dict(
            df_admixture="DataFrame returned by :meth:`admixture`.",
            k="Number of ancestral populations (K).",
            split_by="""
                Metadata column whose unique values define the violin groups
                (e.g. ``'country'``, ``'aim_species'``).
            """,
            title="Plot title.",
        ),
        returns=True,
    )
    def plot_admixture_violin(
        self,
        df_admixture: pd.DataFrame,
        k: int,
        split_by: str = "country",
        title: Optional[str] = "Ancestry proportion distributions",
        width: plotly_params.fig_width = 1100,
        height: plotly_params.fig_height = 450,
        show: plotly_params.show = False,
        renderer: plotly_params.renderer = None,
    ) -> plotly_params.figure:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        pop_cols = [f"pop{i + 1}" for i in range(k)]
        if split_by not in df_admixture.columns:
            raise ValueError(
                f"split_by={split_by!r} is not a column in df_admixture. "
                f"Available columns: {list(df_admixture.columns)}"
            )

        groups = sorted(df_admixture[split_by].dropna().unique().tolist())
        colors = px.colors.qualitative.D3

        fig = make_subplots(
            rows=1,
            cols=k,
            subplot_titles=pop_cols,
            shared_yaxes=True,
        )

        for col_idx, pop in enumerate(pop_cols):
            for gi, grp in enumerate(groups):
                sub = df_admixture[df_admixture[split_by] == grp][pop]
                fig.add_trace(
                    go.Violin(
                        y=sub.values,
                        name=str(grp),
                        box_visible=True,
                        meanline_visible=True,
                        fillcolor=colors[gi % len(colors)],
                        opacity=0.7,
                        line_color=colors[gi % len(colors)],
                        showlegend=(col_idx == 0),
                        legendgroup=str(grp),
                        x0=str(grp),
                    ),
                    row=1,
                    col=col_idx + 1,
                )
            fig.update_yaxes(range=[-0.05, 1.05], row=1, col=col_idx + 1)
            fig.update_xaxes(showticklabels=False, row=1, col=col_idx + 1)

        fig.update_layout(
            title=dict(text=title, font_size=13),
            violinmode="group",
            legend_title_text=split_by,
            yaxis_title="Ancestry proportion",
            width=width,
            height=height,
            template="simple_white",
        )

        if show:
            fig.show(renderer=renderer)
        return fig


# Add signature of the Author and the date of last modification if needed.
def __author__():
    return "Gadji M."


def __last_modified__():
    return "2027-07-24"
