"""Switching linear-Gaussian regression (input-driven HMM), fit by EM.

Memory note. An earlier version padded the design matrix to (S, Tmax, P),
which is fine for a handful of short sessions and fatal at scale: 200
sessions x 30,000 frames x 119 predictors x 8 bytes is ~570 GB. Here the
design and signals stay FLAT (T_total x P); only the per-state emission
log-densities and posteriors are padded, and those are padded in
length-bucketed batches so the padding waste stays bounded. Peak extra
memory is O(T_total * K) plus one batch of O(S_b * Tmax_b * K).
"""
from __future__ import annotations

import numpy as np

from .config import FitConfig


def _lse(a: np.ndarray, axis: int) -> np.ndarray:
    """log-sum-exp with the max trick. Inlined rather than scipy's, whose
    per-call overhead dominates a T-step recursion."""
    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    return (m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))
            ).squeeze(axis)


class SwitchingGLM:
    """Input-driven HMM with linear-Gaussian emissions.

        z_t | z_{t-1} ~ Categorical(A[z_{t-1}, :])
        y_t | z_t = k  ~ N(W_k x_t + b_k, S_k)

    Univariate (D=1) or multivariate (one latent state shared across regions,
    which is the interesting case: it asks whether the recorded set switches
    regime together).

    Fit by EM. The M-step for W is ridge-penalized weighted least squares;
    the bias is unpenalized. Sessions are independent sequences.

    K=1 reduces exactly to ridge regression. It is the baseline the switching
    model must beat on held-out log-likelihood to be worth reporting.
    """

    #: cap on S_b * Tmax_b per forward-backward batch (elements, not bytes)
    MAX_BATCH_ELEMS = 2_000_000

    def __init__(self, cfg: FitConfig | None = None):
        self.cfg = FitConfig() if cfg is None else cfg
        self.K = self.cfg.n_states
        self.A = self.pi = self.W = self.b = self.S = None
        self.ll_trace_: list[float] = []
        self.ll_: float = np.nan
        self.converged_: bool = False
        self.W_animal: dict = {}      # animal -> [W_k]; empty = complete pooling
        self.b_animal: dict = {}

    # ------------------------------------------------------------ batching --
    @classmethod
    def _batches(cls, ss, max_elems: int | None = None) -> list[dict]:
        """Group sessions into length-bucketed batches of frame INDICES.

        Sorting by length before grouping keeps padding waste small: a batch
        of similar-length sessions pads to near their common length rather
        than to the global maximum.
        """
        max_elems = cls.MAX_BATCH_ELEMS if max_elems is None else max_elems
        seqs = sorted((idx for idx, _, _ in ss.sequences()),
                      key=len, reverse=True)
        out, cur = [], []
        for idx in seqs:
            trial = cur + [idx]
            if cur and len(trial) * len(trial[0]) > max_elems:
                out.append(cur)
                cur = [idx]
            else:
                cur = trial
        if cur:
            out.append(cur)

        packed = []
        for group in out:
            lens = np.array([len(i) for i in group])
            S, Tm = len(group), int(lens.max())
            idxm = np.full((S, Tm), -1, int)
            for s, i in enumerate(group):
                idxm[s, :len(i)] = i
            packed.append(dict(idx=idxm, lens=lens, S=S, Tm=Tm,
                               valid=np.arange(Tm)[None, :] < lens[:, None]))
        return packed

    # ---------------------------------------------------------- emissions --
    def _state_mean(self, ss, k: int) -> np.ndarray:
        """(T, D) mean for state k, using per-animal weights where the animal
        was seen during fitting and the group weights otherwise."""
        if not getattr(self, 'W_animal', None):
            return ss.X @ self.W[k].T + self.b[k]
        out = ss.X @ self.W[k].T + self.b[k]
        for a, Ws in self.W_animal.items():
            m = ss.animal == a
            if m.any() and Ws[k] is not None:
                out[m] = ss.X[m] @ Ws[k].T + self.b_animal[a][k]
        return out

    def _log_emission_flat(self, ss) -> np.ndarray:
        """(T_total, K) log N(y_t | W_k x_t + b_k, S_k), computed on the flat
        design. No padding -- this is the large array and it stays flat."""
        Y = ss.Y
        D = Y.shape[1]
        out = np.empty((len(ss.X), self.K))
        for k in range(self.K):
            R = Y - self._state_mean(ss, k)
            Sk = self.S[k] + self.cfg.var_floor * np.eye(D)
            if D == 1:
                v = max(float(Sk[0, 0]), self.cfg.var_floor)
                out[:, k] = -0.5 * (np.log(2 * np.pi * v) + R[:, 0] ** 2 / v)
            else:
                try:
                    L = np.linalg.cholesky(Sk)
                except np.linalg.LinAlgError:
                    w, V = np.linalg.eigh(Sk)
                    L = np.linalg.cholesky(
                        V @ np.diag(np.maximum(w, self.cfg.var_floor)) @ V.T)
                sol = np.linalg.solve(L, R.T).T
                out[:, k] = -0.5 * (D * np.log(2 * np.pi)
                                    + 2.0 * np.log(np.diag(L)).sum()
                                    + (sol ** 2).sum(1))
        return out

    @staticmethod
    def _gather(flat: np.ndarray, bt: dict) -> np.ndarray:
        """Scatter flat per-frame values into this batch's padded layout."""
        g = np.zeros((bt['S'], bt['Tm'], flat.shape[1]))
        m = bt['valid']
        g[m] = flat[bt['idx'][m]]
        return g

    # ------------------------------------------ forward-backward (batched) --
    def _fb_batch(self, logB: np.ndarray, bt: dict):
        S, Tm, K = bt['S'], bt['Tm'], self.K
        lens, valid = bt['lens'], bt['valid']
        logA = np.log(self.A + 1e-300)

        la = np.full((S, Tm, K), -np.inf)
        la[:, 0, :] = np.log(self.pi + 1e-300) + logB[:, 0, :]
        for t in range(1, Tm):
            m = lens > t
            if not m.any():
                break
            la[m, t, :] = _lse(la[m, t - 1, :][:, :, None] + logA[None], 1) \
                + logB[m, t, :]

        lb = np.zeros((S, Tm, K))
        for t in range(Tm - 2, -1, -1):
            m = lens > t + 1
            if not m.any():
                continue
            lb[m, t, :] = _lse(
                logA[None] + (logB[m, t + 1, :] + lb[m, t + 1, :])[:, None, :], 2)

        ll_seq = _lse(la[np.arange(S), lens - 1, :], axis=1)

        lg = la + lb - ll_seq[:, None, None]
        lg[~valid] = -np.inf
        g = np.exp(lg - _lse(lg, axis=2)[..., None])
        g[~valid] = 0.0

        xi = np.zeros((K, K))
        chunk = max(1, int(4e6 / max(S * K * K, 1)))
        for t0 in range(0, max(Tm - 1, 0), chunk):
            t1 = min(t0 + chunk, Tm - 1)
            if t1 <= t0:
                break
            e = np.exp(la[:, t0:t1, :, None] + logA[None, None]
                       + (logB[:, t0 + 1:t1 + 1, :]
                          + lb[:, t0 + 1:t1 + 1, :])[:, :, None, :]
                       - ll_seq[:, None, None, None])
            e[~valid[:, t0 + 1:t1 + 1]] = 0.0
            xi += e.sum((0, 1))
        return g, xi, float(ll_seq.sum())

    def _e_step(self, ss, batches, logB_flat):
        """Returns flat posteriors (T_total, K), summed xi, total LL, and the
        summed first-frame posteriors for pi."""
        gamma = np.zeros((len(ss.X), self.K))
        xi = np.zeros((self.K, self.K))
        p0 = np.zeros(self.K)
        ll = 0.0
        for bt in batches:
            g, x, l = self._fb_batch(self._gather(logB_flat, bt), bt)
            m = bt['valid']
            gamma[bt['idx'][m]] = g[m]
            xi += x
            p0 += g[:, 0, :].sum(0)
            ll += l
        return gamma, xi, p0, ll

    # --------------------------------------------------------- M-step bits --
    def _ridge_solve(self, G, C, P, lam):
        Reg = lam * np.eye(P + 1)
        Reg[-1, -1] = 0.0                        # bias unpenalized
        return (lambda beta: (beta[:P].T, beta[P]))(np.linalg.solve(G + Reg, C))

    def _solve_weights_hier(self, ss, gamma, k, animals, idx_by_animal):
        """Partial-pooling weight update for one state.

        Fits W_a = W_group + delta_a with ridge `ridge` on W_group and
        `hier_tau` on each delta_a. Solved as one joint system in
        [beta_group; delta_1 .. delta_A] whose blocks are per-animal Gram
        matrices, so nothing of size (T x A*P) is ever materialised:

            row g : (sum_a G_a + lam I) beta + sum_a G_a delta_a = sum_a C_a
            row a : G_a beta + (G_a + tau I) delta_a             = C_a

        Returns (W_group, b_group, {animal: (W_a, b_a)}).
        """
        P, D = ss.X.shape[1], ss.Y.shape[1]
        A, Q = len(animals), P + 1
        lam, tau = self.cfg.ridge, float(self.cfg.hier_tau)

        Gs, Cs = [], []
        for a in animals:
            i = idx_by_animal[a]
            Xa = np.column_stack([ss.X[i], np.ones(i.size)])
            w = gamma[i, k]
            Xw = Xa * w[:, None]
            Gs.append(Xw.T @ Xa)
            Cs.append(Xw.T @ ss.Y[i])

        M = np.zeros(((A + 1) * Q, (A + 1) * Q))
        R = np.zeros(((A + 1) * Q, D))
        Gsum = np.zeros((Q, Q))
        for a in range(A):
            Gsum += Gs[a]
            R[:Q] += Cs[a]
            M[:Q, (a + 1) * Q:(a + 2) * Q] = Gs[a]
            M[(a + 1) * Q:(a + 2) * Q, :Q] = Gs[a]
            blk = Gs[a] + tau * np.eye(Q)
            blk[-1, -1] -= tau                        # intercepts unpenalised
            M[(a + 1) * Q:(a + 2) * Q, (a + 1) * Q:(a + 2) * Q] = blk
            R[(a + 1) * Q:(a + 2) * Q] = Cs[a]
        Reg = lam * np.eye(Q)
        Reg[-1, -1] = 0.0
        M[:Q, :Q] = Gsum + Reg

        beta = np.linalg.solve(M + 1e-8 * np.eye(M.shape[0]), R)
        Wg, bg = beta[:P].T, beta[P]
        per = {}
        for a_i, a in enumerate(animals):
            d = beta[(a_i + 1) * Q:(a_i + 2) * Q]
            per[a] = (Wg + d[:P].T, bg + d[P])
        return Wg, bg, per

    def _init(self, ss, rng):
        X, Y = ss.X, ss.Y
        P, D = X.shape[1], Y.shape[1]
        Xb = np.column_stack([X, np.ones(len(X))])
        W0, b0 = self._ridge_solve(Xb.T @ Xb, Xb.T @ Y, P, self.cfg.ridge)
        R = Y - (X @ W0.T + b0)
        self.W = np.stack([W0 * (1.0 + 0.15 * rng.standard_normal(W0.shape))
                           for _ in range(self.K)])
        self.b = np.stack([b0 + 0.15 * rng.standard_normal(D) * R.std(0)
                           for _ in range(self.K)])
        s = np.atleast_2d(np.cov(R.T)) if D > 1 else np.array([[R.var()]])
        self.S = np.stack([s * (0.5 + rng.random()) for _ in range(self.K)])
        A = np.full((self.K, self.K), 1.0 / max(self.K, 1))
        np.fill_diagonal(A, 0.9)
        self.A = A / A.sum(1, keepdims=True)
        self.pi = np.full(self.K, 1.0 / self.K)

    # ----------------------------------------------------------------- fit --
    def fit(self, ss, verbose: bool | None = None) -> 'SwitchingGLM':
        cfg = self.cfg
        verbose = cfg.verbose if verbose is None else verbose
        batches = self._batches(ss)
        P, D = ss.X.shape[1], ss.Y.shape[1]
        Xb = np.column_stack([ss.X, np.ones(len(ss.X))])
        Y = ss.Y
        animals = sorted(np.unique(ss.animal).tolist())
        idx_by_animal = {a: np.flatnonzero(ss.animal == a) for a in animals}
        self.W_animal, self.b_animal = {}, {}

        best_ll, best = -np.inf, None
        for r in range(cfg.n_restarts):
            rng = np.random.default_rng(cfg.seed + 1000 * r)
            self.W_animal, self.b_animal = {}, {}
            self._init(ss, rng)
            prev, trace, converged, it, ll = -np.inf, [], False, 0, -np.inf

            for it in range(cfg.max_iter):
                gamma, xi, p0, ll = self._e_step(
                    ss, batches, self._log_emission_flat(ss))

                if cfg.share_weights:
                    # One W pooled across states; bias stays state-specific.
                    # Solve the joint problem with a shared slope block and K
                    # free intercepts, so b_k is not absorbed into W.
                    G = np.zeros((P + self.K, P + self.K))
                    C = np.zeros((P + self.K, D))
                    XtX = ss.X.T @ ss.X
                    G[:P, :P] = XtX
                    C[:P] = ss.X.T @ Y
                    for k in range(self.K):
                        w = gamma[:, k]
                        xw = ss.X.T @ w
                        G[:P, P + k] = xw
                        G[P + k, :P] = xw
                        G[P + k, P + k] = w.sum()
                        C[P + k] = w @ Y
                    # gamma rows sum to 1, so the shared slope block is the
                    # unweighted normal equation; only the intercepts are
                    # responsibility-weighted.
                    Reg = cfg.ridge * np.eye(P + self.K)
                    Reg[P:, P:] = 0.0                   # intercepts free
                    beta = np.linalg.solve(G + Reg, C)
                    Wshared = beta[:P].T
                    for k in range(self.K):
                        self.W[k] = Wshared
                        self.b[k] = beta[P + k]
                elif cfg.hier_tau is not None:
                    # Partial pooling: W_a = W_group + delta_a, each delta
                    # penalised by hier_tau. Held-out animals fall back to
                    # W_group, which this estimates more cleanly than complete
                    # pooling does when filters are animal-specific.
                    for k in range(self.K):
                        Wg, bg, per = self._solve_weights_hier(
                            ss, gamma, k, animals, idx_by_animal)
                        self.W[k], self.b[k] = Wg, bg
                        for a, (Wa, ba) in per.items():
                            self.W_animal.setdefault(a, [None] * self.K)
                            self.b_animal.setdefault(a, [None] * self.K)
                            self.W_animal[a][k] = Wa
                            self.b_animal[a][k] = ba
                else:
                    for k in range(self.K):               # weights
                        Xw = Xb * gamma[:, k][:, None]
                        self.W[k], self.b[k] = self._ridge_solve(
                            Xw.T @ Xb, Xw.T @ Y, P, cfg.ridge)

                for k in range(self.K):                   # covariances
                    Rk = Y - self._state_mean(ss, k)
                    w = gamma[:, k]
                    self.S[k] = ((Rk * w[:, None]).T @ Rk / max(w.sum(), 1.0)
                                 + cfg.var_floor * np.eye(D))

                A = np.maximum(xi + cfg.sticky * np.eye(self.K), 1e-12)
                self.A = A / A.sum(1, keepdims=True)
                self.pi = p0 / max(p0.sum(), 1e-12)

                trace.append(ll)
                if abs(ll - prev) < cfg.tol * max(abs(prev), 1.0):
                    converged = True
                    break
                prev = ll

            if verbose:
                print(f'   restart {r}: ll={ll:.1f} '
                      f'({it + 1} iters, converged={converged})', flush=True)
            if ll > best_ll:
                best_ll = ll
                best = (self.A.copy(), self.pi.copy(), self.W.copy(),
                        self.b.copy(), self.S.copy(), list(trace), converged,
                        {a: list(v) for a, v in self.W_animal.items()},
                        {a: list(v) for a, v in self.b_animal.items()})

        (self.A, self.pi, self.W, self.b, self.S, self.ll_trace_,
         self.converged_, self.W_animal, self.b_animal) = best
        self.ll_ = float(best_ll)
        self._order_states(ss)
        return self

    def _order_states(self, ss) -> None:
        """Canonical state order by occupancy, descending.

        Label switching is inherent to EM; without a canonical order, weight
        tables are not comparable across restarts, folds, or regions.
        """
        o = np.argsort(-self.posterior_occupancy(ss))
        self.W, self.b, self.S = self.W[o], self.b[o], self.S[o]
        self.A = self.A[np.ix_(o, o)]
        self.pi = self.pi[o]

    # ------------------------------------------------------------ inference
    def posteriors(self, ss) -> np.ndarray:
        """Smoothed state posteriors gamma[t, k], in original frame order.

        Uses future observations -- a fit statistic, not a prediction.
        """
        return self._e_step(ss, self._batches(ss),
                            self._log_emission_flat(ss))[0]

    def posterior_occupancy(self, ss) -> np.ndarray:
        return self.posteriors(ss).mean(0)

    def log_likelihood(self, ss) -> float:
        return self._e_step(ss, self._batches(ss),
                            self._log_emission_flat(ss))[3]

    def filtered_predict(self, ss) -> np.ndarray:
        """One-step-ahead prediction under FILTERED (causal) state beliefs.

        Smoothed posteriors see the future, so a state-conditioned R^2 built
        from them is circular. Held-out performance must use this.
        """
        logB_flat = self._log_emission_flat(ss)
        logA = np.log(self.A + 1e-300)
        D = ss.Y.shape[1]
        mu_flat = np.stack([self._state_mean(ss, k)
                            for k in range(self.K)], axis=1)   # (T, K, D)
        out = np.zeros_like(ss.Y)
        for bt in self._batches(ss):
            logB = self._gather(logB_flat, bt)
            S, Tm = bt['S'], bt['Tm']
            mu = np.zeros((S, Tm, self.K, D))
            m = bt['valid']
            mu[m] = mu_flat[bt['idx'][m]]
            Yp = np.zeros((S, Tm, D))
            la = np.tile(np.log(self.pi + 1e-300), (S, 1))
            for t in range(Tm):
                pred = la if t == 0 else _lse(la[:, :, None] + logA[None], 1)
                p = np.exp(pred - _lse(pred, axis=1)[:, None])
                Yp[:, t, :] = np.einsum('sk,skd->sd', p, mu[:, t])
                la = pred + logB[:, t, :]
                la = la - _lse(la, axis=1)[:, None]
            out[bt['idx'][m]] = Yp[m]
        return out

    def stationary(self) -> np.ndarray:
        """Stationary state distribution: leading left eigenvector of A."""
        w, V = np.linalg.eig(self.A.T)
        v = np.real(V[:, int(np.argmin(np.abs(w - 1.0)))])
        v = np.abs(v)
        return v / v.sum()

    def input_predict(self, ss) -> np.ndarray:
        """Prediction under the STATIONARY state distribution.

        y_hat_t = sum_k pi_stat[k] (W_k x_t + b_k)

        Conditions on nothing from y, so the resulting R^2 measures only what
        the regressors predict and is directly comparable across K.

        `filtered_predict` instead mixes by p(z_t | y_1..y_{t-1}), which is a
        proper one-step-ahead forecast but is a function of the signal's own
        past: with a state-specific bias and a signal autocorrelated over
        seconds, that posterior acts as an adaptive estimate of the local
        level of y. That channel is why predictive R^2 climbs with K whether
        or not any behavioural code exists. Report both.

        IMPORTANT LIMIT. Because this averages W_k over the stationary
        distribution, two genuinely different state weight sets collapse to
        their average -- which is close to the K=1 solution. So this metric
        is FLAT ACROSS K whether or not state-dependent coding is real, and
        verified so in both directions on synthetic data. It is the
        across-K-comparable encoding estimate, and it correctly refuses to
        inflate; it is NOT a test for state-dependent coding. For that, use
        evaluate.weight_sharing_test (held-out likelihood, unconstrained vs
        pooled W at matched K) and the per-state weight table.
        """
        pi = self.stationary()
        out = np.zeros_like(ss.Y)
        for k in range(self.K):
            out += pi[k] * self._state_mean(ss, k)
        return out

    def viterbi(self, ss) -> np.ndarray:
        """MAP state path per frame, in original frame order."""
        logB_flat = self._log_emission_flat(ss)
        logA = np.log(self.A + 1e-300)
        out = np.zeros(len(ss.X), int)
        for bt in self._batches(ss):
            logB = self._gather(logB_flat, bt)
            S, Tm, lens = bt['S'], bt['Tm'], bt['lens']
            d = np.full((S, Tm, self.K), -np.inf)
            psi = np.zeros((S, Tm, self.K), int)
            d[:, 0, :] = np.log(self.pi + 1e-300) + logB[:, 0, :]
            for t in range(1, Tm):
                m = lens > t
                if not m.any():
                    break
                z = d[m, t - 1, :][:, :, None] + logA[None]
                psi[m, t, :] = z.argmax(1)
                d[m, t, :] = z.max(1) + logB[m, t, :]
            path = np.zeros((S, Tm), int)
            for s in range(S):
                n = int(lens[s])
                path[s, n - 1] = int(d[s, n - 1].argmax())
                for t in range(n - 2, -1, -1):
                    path[s, t] = psi[s, t + 1, path[s, t + 1]]
            m = bt['valid']
            out[bt['idx'][m]] = path[m]
        return out

    # ----------------------------------------------------------- properties
    @property
    def n_params(self) -> int:
        K = self.K
        D, P = self.W.shape[1], self.W.shape[2]
        return K * (P * D + D + D * (D + 1) // 2) + K * (K - 1) + (K - 1)

    def expected_dwell_s(self, fs: float = 40.0) -> np.ndarray:
        """Geometric mean dwell implied by the self-transitions.

        Dwell under ~1 s is not credible given GCaMP6f kinetics; raise
        FitConfig.sticky if states flicker at the frame scale.
        """
        return 1.0 / ((1.0 - np.diag(self.A) + 1e-12) * fs)

    def to_dict(self) -> dict:
        """Serializable parameter bundle (no SessionSet reference)."""
        return dict(K=self.K, A=self.A, pi=self.pi, W=self.W, b=self.b,
                    S=self.S, ll=self.ll_, ll_trace=np.asarray(self.ll_trace_),
                    converged=self.converged_,
                    W_animal={a: np.stack(v) for a, v in self.W_animal.items()},
                    b_animal={a: np.stack(v) for a, v in self.b_animal.items()},
                    cfg=vars(self.cfg).copy()
                    if hasattr(self.cfg, '__dict__') else dict())

    @classmethod
    def from_dict(cls, d: dict, cfg: FitConfig | None = None) -> 'SwitchingGLM':
        m = cls(cfg or FitConfig(n_states=int(d['K'])))
        m.K = int(d['K'])
        m.A, m.pi, m.W, m.b, m.S = d['A'], d['pi'], d['W'], d['b'], d['S']
        m.ll_ = float(d['ll'])
        m.ll_trace_ = list(np.asarray(d['ll_trace']))
        m.converged_ = bool(d['converged'])
        m.W_animal = {a: list(v) for a, v in d.get('W_animal', {}).items()}
        m.b_animal = {a: list(v) for a, v in d.get('b_animal', {}).items()}
        return m
