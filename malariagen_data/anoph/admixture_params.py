"""Type aliases and defaults for admixture analysis parameters."""

import pandas as pd
from typing_extensions import Annotated, TypeAlias

from malariagen_data.anoph import base_params  # noqa: F401  (re-exported)


k: TypeAlias = Annotated[
    int,
    """
    Number of hypothetical ancestral populations (K).  Must be ≥ 2.
    Typical values for the *An. gambiae* complex are 2–5; higher values
    increase compute time and may over-fit on small cohorts.
    """,
]

k_default: k = 2

imputation_method: TypeAlias = Annotated[
    str,
    """
    Strategy for imputing missing genotype calls (sentinel ``-127``) before
    NMF:

    * ``'mean'`` *(default)* — replace each missing dosage with the
      per-site column mean.  Preserves the marginal allele-frequency
      distribution and is the recommended choice.
    * ``'zero'`` — replace missing values with 0 (homozygous reference).
      Conservative; may under-estimate heterozygous ancestry in samples
      with high missingness.
    """,
]

imputation_method_default: imputation_method = "mean"

max_iter: TypeAlias = Annotated[
    int,
    """
    Maximum number of NMF multiplicative-update iterations.  Convergence
    is detected by monitoring the change in root-mean-squared reconstruction
    error (RMSE) every 10 iterations.  Most analyses converge in 50–200
    iterations; the default of 300 provides a safety margin without
    excessive runtime.
    """,
]

max_iter_default: max_iter = 300

df_admixture: TypeAlias = Annotated[
    pd.DataFrame,
    """
    A DataFrame with one row per sample and the following columns:

    * ``sample_id`` — sample identifier.
    * ``pop1`` … ``popK`` — estimated ancestry proportion from each of the
      K hypothetical ancestral populations.  Values are non-negative and
      each row sums to 1.0.
    * Sample metadata columns joined from ``sample_metadata()`` (e.g.
      ``country``, ``location``, ``taxon``, ``year``).
    """,
]
