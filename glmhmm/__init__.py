"""glmhmm -- switching linear-Gaussian regression (input-driven HMM) for
multi-fiber photometry.

Falkner lab / observational aggression. Fits

    z_t | z_{t-1} ~ Categorical(A[z_{t-1}, :])
    y_t | z_t = k  ~ N(W_k x_t + b_k, S_k)

where x_t stacks temporally-filtered dyad kinematics, attack-event
regressors, one-hot observer-behavior group, and autoregressive lags of y.

NOMENCLATURE. This is NOT the Ashwood/Pillow GLM-HMM, which has
Bernoulli/categorical emissions over discrete choices. Emissions here are
continuous Gaussian. In a methods section call it a switching linear-Gaussian
regression, or an input-driven HMM with Gaussian emissions.

Typical use
-----------
    from glmhmm import *
    ss = build_sessionset(keys, full_data, features, labels, REGIONS)
    allr, summ = select_K(ss, Ks=(1, 2, 3, 4))
    m = SwitchingGLM(FitConfig(n_states=3)).fit(ss)

K=1 reduces exactly to ridge regression and is the mandatory baseline.

env: postdoc-env (Windows). numpy, scipy, pandas, scikit-learn, matplotlib.
"""
from .config import (DesignConfig, FitConfig, BEHAVIOR_GROUPS, OTHER_GROUP,
                     BEHAVIOR_REFERENCE, OKABE_ITO, set_nature_rc)
from .data import (parse_session, audit_lengths, audit_alignment, effective_n,
                   audit_effective_n, behavior_onehot, feature_group, align_labels,
                   SessionSet, build_sessionset)
from .model import SwitchingGLM
from .evaluate import (fit_loao, select_K, state_behavior_correspondence,
                       state_summary, weight_table, shuffle_control,
                       variance_partition, weight_sharing_test, fit_loso_within_animal,
                       scaling_stability,
                       states_are_variance_regimes)
from .plotting import (plot_K_selection, plot_state_raster, plot_weight_matrix,
                       plot_state_behavior, plot_loao_r2, savefig)
from . import demo

__all__ = [
    'DesignConfig', 'FitConfig', 'BEHAVIOR_GROUPS', 'OTHER_GROUP',
    'BEHAVIOR_REFERENCE', 'OKABE_ITO', 'set_nature_rc',
    'parse_session', 'audit_lengths', 'audit_alignment', 'effective_n',
    'audit_effective_n', 'behavior_onehot', 'feature_group',
    'align_labels',
    'SessionSet', 'build_sessionset', 'SwitchingGLM',
    'fit_loao', 'select_K', 'state_behavior_correspondence', 'state_summary',
    'weight_table', 'shuffle_control', 'variance_partition', 'weight_sharing_test',
    'fit_loso_within_animal',
    'scaling_stability',
    'states_are_variance_regimes',
    'plot_K_selection', 'plot_state_raster', 'plot_weight_matrix',
    'plot_state_behavior', 'plot_loao_r2', 'savefig', 'demo',
]
__version__ = '0.1.0'
