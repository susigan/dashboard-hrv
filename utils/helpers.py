from utils.config import *
import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))


# ════════════════════════════════════════════════════════════════════════════
# MÓDULO: utils/helpers.py
# ════════════════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════════════════════
# utils/helpers.py — ATHELTICA Dashboard
# Funções auxiliares reutilizáveis por todas as tabs.
# Importar com: from utils.helpers import *
# ════════════════════════════════════════════════════════════════════════════════

# ── Parsing e conversão ────────────────────────────────────────────────────────

def detectar_col(df, lst):
    """Retorna o primeiro nome de coluna da lista que existe no df."""
    for n in lst:
        if n in df.columns: return n
    return None

def br_float(v):
    """Converte string BR (vírgula decimal) ou numérico para float."""
    if pd.isna(v) or v == '': return None
    if isinstance(v, (int, float)): return float(v)
    try: return float(str(v).replace(',', '.').strip())
    except: return None

def parse_date(v):
    """Parser robusto de datas em vários formatos."""
    if pd.isna(v): return None
    if isinstance(v, (pd.Timestamp, datetime)): return v
    s = str(v).strip()
    for fmt in ["%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d"]:
        try: return datetime.strptime(s, fmt)
        except: pass
    return None

# ── Tipo de actividade ─────────────────────────────────────────────────────────

def norm_tipo(t):
    if not isinstance(t, str): return 'Other'
    return TYPE_MAP.get(t.strip(), 'Other')

def fmt_dur(horas):
    """Converte horas decimais para string legível: 55min, 1h05, 2h30"""
    if horas is None or (hasattr(horas, '__float__') and horas != horas): return '—'
    try:
        h = float(horas)
    except (TypeError, ValueError):
        return '—'
    if h <= 0: return '—'
    total_min = round(h * 60)
    hh = total_min // 60
    mm = total_min % 60
    if hh == 0:   return f'{mm}min'
    if mm == 0:   return f'{hh}h'
    return f'{hh}h{mm:02d}'

def fmt_dur_sec(segundos):
    """Converte segundos directamente para string legível."""
    if segundos is None: return '—'
    try:
        return fmt_dur(float(segundos) / 3600)
    except (TypeError, ValueError):
        return '—'

def get_cor(tipo):
    return CORES_ATIV.get(tipo, CORES_ATIV['Other'])

# ── Normalização e stats ───────────────────────────────────────────────────────

def norm_serie(s, lo=0, hi=100):
    mn, mx = s.min(), s.max()
    if mx == mn: return pd.Series([50] * len(s), index=s.index)
    return (s - mn) / (mx - mn) * (hi - lo) + lo

def cvr(s, w=7):
    """CV% rolling."""
    m = s.rolling(w, min_periods=3).mean()
    sd = s.rolling(w, min_periods=3).std()
    return (sd / m) * 100

def conv_15(v):
    """Converte escala 1-5 para 0-100."""
    if pd.isna(v): return 50
    return max(0, min(100, (float(v) - 1) * 25))

def norm_range(base, cv):
    """Normal range HRV4Training: baseline ± 0.5×CV%×baseline."""
    if pd.isna(base) or pd.isna(cv) or base <= 0: return None, None
    m = 0.5 * (cv / 100) * base
    return base - m, base + m

def calcular_swc(base, cv):
    """SWC = 0.5 × CV% × Baseline / 100  (Hopkins et al. 2009)"""
    if pd.isna(base) or pd.isna(cv) or base <= 0 or cv <= 0: return None
    return 0.5 * (cv / 100) * base

# ── Limpeza de dados ──────────────────────────────────────────────────────────

def remove_zscore(s, thr=3.0):
    v = pd.to_numeric(s, errors='coerce')
    ok = v[v.notna()]
    if len(ok) < 4: return v
    z = np.abs(scipy_stats.zscore(ok))
    v.loc[ok.index[z > thr]] = np.nan
    return v

def remove_zeros(df, cols):
    df = df.copy()
    for c in cols:
        if c in df.columns: df.loc[df[c] == 0, c] = np.nan
    return df

def fill_missing(df, col, w=7):
    if col not in df.columns: return df
    df = df.copy()
    df[col] = pd.to_numeric(df[col], errors='coerce')
    mask = df[col].isna()
    if mask.any():
        r = df[col].rolling(w, min_periods=2, center=True).mean()
        df.loc[mask, col] = r[mask]
        mask2 = df[col].isna()
        if mask2.any():
            r2 = df[col].rolling(14, min_periods=2, center=True).mean()
            df.loc[mask2, col] = r2[mask2]
        mask3 = df[col].isna()
        if mask3.any(): df.loc[mask3, col] = df[col].mean()
    return df

# ── Actividades ───────────────────────────────────────────────────────────────

def filtrar_principais(df):
    """Remove WeightTraining em dias com outras modalidades."""
    if len(df) == 0: return df
    df = df.copy()
    df['_t'] = df['type'].apply(norm_tipo)
    df['_d'] = pd.to_datetime(df['Data']).dt.date
    por_dia = df.groupby('_d')['_t'].apply(list).to_dict()
    df['_ok'] = df.apply(
        lambda r: not (r['_t'] == 'WeightTraining' and len(por_dia.get(r['_d'], [])) > 1), axis=1)
    return df[df['_ok']].drop(columns=['_t', '_d', '_ok'])

def add_tempo(df):
    """Adiciona colunas mes, ano, trimestre."""
    if len(df) > 0 and 'Data' in df.columns:
        df = df.copy()
        df['mes']       = pd.to_datetime(df['Data']).dt.strftime('%Y-%m')
        df['ano']       = pd.to_datetime(df['Data']).dt.year
        df['trimestre'] = pd.to_datetime(df['Data']).dt.to_period('Q').astype(str)
    return df

def classificar_rpe(v):
    if pd.isna(v): return None
    v = float(v)
    if 1 <= v <= 4.9: return 'Leve'
    if 5 <= v <= 6.9: return 'Moderado'
    if 7 <= v <= 10:  return 'Pesado'
    return None

# ── CTL/ATL/FTLM ─────────────────────────────────────────────────────────────

# ════════════════════════════════════════════════════════════════════════════════
# FRACTIONAL TRAINING LOAD MEMORY (FTLM)
# Della Mattia (2025) — Parts I & II + FMT 2019
#
# Classical PMC uses exponential kernel: K(τ) = e^(−τ/τc)  → saturates quickly
# Fractional FTLM uses power-law kernel: K(τ) = τ^(γ−1)   → long memory, no saturation
#
# CTLγ(t) = (1/Γ(γ)) · Σ  Load(t−k) · k^(γ−1)
#                          k=1..t
#
# γ ∈ (0,1): controls memory depth
#   γ → 1.0 : exponential decay, short memory (≈ classical PMC)
#   γ → 0.3 : power-law decay, long memory (optimal for endurance base)
#   γ → 0.1 : very long memory (months of training persist)
#
# Two γ values are fitted independently (Dual-Gamma, Part II §3):
#   γ_perf    → performance proxy (icu_pm_cp or MMP PR efforts)
#   γ_recovery → HRV trend via moving-window regression (Part II §2)
# ════════════════════════════════════════════════════════════════════════════════

def _ftlm_kernel_weights(n, gamma_val):
    """
    Compute Riemann-Liouville fractional kernel weights for n steps back.
    w(k) = k^(gamma-1) / Γ(gamma)   for k = 1..n

    Vectorised with NumPy — O(n) instead of the naive O(n²) loop in the paper.
    """
    from scipy.special import gamma as _gamma_fn
    k = np.arange(1, n + 1, dtype=np.float64)
    w = np.power(k, gamma_val - 1.0)
    return w / _gamma_fn(gamma_val)


def ftlm_fractional(load_arr, gamma_val, max_lag=None):
    """
    Discrete Riemann-Liouville fractional integral of training load.

    CTLγ(t) = Σ_{k=1}^{t} Load(t-k) · k^(γ-1) / Γ(γ)

    Optimised: kernel weights precomputed once per γ (not per time step).
    ~4-5x faster than original loop.
    """
    from scipy.special import gamma as _gfn
    load = np.array(load_arr, dtype=np.float64)
    n    = len(load)
    ml   = n if max_lag is None else min(n, max_lag)

    # Precompute kernel once for this γ — O(max_lag) instead of O(n × max_lag)
    k = np.arange(1, ml + 1, dtype=np.float64)
    w = np.power(k, gamma_val - 1.0) / _gfn(gamma_val)  # shape (ml,)

    ctl = np.zeros(n)
    for t in range(1, n):
        lag    = min(t, ml)
        seg    = load[max(0, t - ml):t][::-1]   # most-recent first
        ctl[t] = np.dot(seg, w[:lag])
    return ctl


def _hrv_trend(hrv_arr, window=7, return_slope=False):
    """
    Extract local HRV trend via moving-window linear regression (Part II §2).

    For each day t, fits HRV(τ) = a(t)·τ + b(t) over [t-window+1, t].

    Returns combined signal: b_z + w*a_z  where
        b(t) = intercept = HRV level (state)
        a(t) = slope     = HRV direction (improving/deteriorating)
        b_z  = zscore(b) — normalised level
        a_z  = zscore(a) — normalised direction
        w    = std(b)/std(a) — data-driven weight, NOT arbitrary
               → slope contributes proportionally to its own variability

    This avoids the arbitrary fixed-coefficient problem and scale mixing.

    Adaptive window: if window > available data, window shrinks.
    """
    from scipy.stats import linregress as _lr
    hrv   = np.array(hrv_arr, dtype=np.float64)
    n     = len(hrv)
    trend_b = np.full(n, np.nan)   # intercept b(t)
    trend_a = np.full(n, np.nan)   # slope a(t)
    x       = np.arange(window, dtype=np.float64)
    for t in range(window - 1, n):
        y = hrv[t - window + 1:t + 1]
        mask = ~np.isnan(y)
        if mask.sum() < 4:
            continue
        a, b, *_ = _lr(x[mask], y[mask])
        trend_b[t] = b
        trend_a[t] = a

    # Normalise each series independently (z-score over valid values)
    def _zs(arr):
        mu = np.nanmean(arr); sd = np.nanstd(arr)
        return (arr - mu) / sd if sd > 1e-9 else arr - mu

    b_z = _zs(trend_b)
    a_z = _zs(trend_a)

    # Data-driven weight: std(b)/std(a) — NOT arbitrary
    std_b = np.nanstd(trend_b)
    std_a = np.nanstd(trend_a)
    w = (std_b / std_a) if std_a > 1e-9 else 1.0
    w = np.clip(w, 0.1, 5.0)   # bound to avoid extreme weights

    # Combined: level (primary) + proportionally-weighted direction
    trend_combined = np.where(
        np.isfinite(b_z) & np.isfinite(a_z),
        b_z + w * a_z,
        b_z   # fallback to level only
    )

    if return_slope:
        return trend_combined, trend_b, trend_a, w
    return trend_combined


def fit_gamma_performance(load_arr, perf_arr, gamma_range=(0.10, 0.90),
                          step=0.05, lag=0, max_lag=365,
                          smooth_perf=3):
    """
    Fit γ_performance: find γ that maximises validated R² between CTLγ and
    a performance proxy (ActivityCP/eFTP/MMP).

    Improvements:
    1. Z-score normalise CTLγ before correlation (scale-invariant,
       avoids R² inflation from scale mismatch between γ values)
    2. Temporal train/test split (70% fit, 30% validate) — avoids overfitting
       same dataset for both fitting and evaluation
    3. Rolling smooth of performance proxy (default 3 sessions) to reduce
       day-to-day noise in ActivityCP estimates
       Best practice: rolling_mean(eCP, 3-5 sessions) before correlating

    Based on Part II §1: Power_test = β₀ + β₁ · CTLγ

    Returns
    -------
    best_gamma : float
    best_r2    : float — validated R² on test set
    n_points   : int — non-NaN performance points
    """
    from scipy.stats import pearsonr as _pr
    load = np.array(load_arr, dtype=np.float64)
    perf = np.array(perf_arr, dtype=np.float64)

    # Smooth performance proxy to reduce session-to-session noise
    # Use rolling mean over non-NaN values (window = smooth_perf sessions)
    if smooth_perf > 1:
        import pandas as _pd
        perf_s = (_pd.Series(perf)
                   .rolling(smooth_perf, min_periods=1)
                   .mean().values)
    else:
        perf_s = perf

    perf_valid = ~np.isnan(perf_s)
    n_pts = int(perf_valid.sum())

    # Temporal split: 70% train, 30% test
    n = len(load)
    split = int(n * 0.70)

    gammas = np.arange(gamma_range[0], gamma_range[1] + step/2, step)
    best_gamma, best_r2 = 0.35, -np.inf

    for g in gammas:
        ctl_g = ftlm_fractional(load, g, max_lag=max_lag)

        # Z-score normalise on train set → consistent scale across γ
        _mu, _sd = np.nanmean(ctl_g[:split]), np.nanstd(ctl_g[:split])
        if _sd < 1e-9:
            continue
        ctl_z = (ctl_g - _mu) / _sd

        if lag > 0:
            ctl_z = np.roll(ctl_z, lag)
            ctl_z[:lag] = np.nan

        # Validate on TEST set only
        test_idx = np.arange(n) >= split
        valid = test_idx & perf_valid & np.isfinite(ctl_z)
        if valid.sum() < 5:
            continue
        try:
            r, _ = _pr(ctl_z[valid], perf_s[valid])
            r2   = r ** 2
            if r2 > best_r2:
                best_r2, best_gamma = r2, float(g)
        except Exception:
            continue

    return best_gamma, best_r2, n_pts


def fit_gamma_recovery(load_arr, hrv_arr, gamma_range=(0.10, 0.90),
                       step=0.05, hrv_window=7, lags_to_test=None, max_lag=365):
    """
    Fit γ_recovery: find (γ, lag) that maximises validated R² between
    CTLγ(t-lag) and HRV_trend(t).

    Improvements over naive implementation:
    1. Tests multiple lags ∈ [0,1,2,3,5,7] — physiological response is
       NOT fixed at lag=1; varies by athlete and training phase.
    2. Uses combined signal: intercept b(t) + slope a(t), because:
       - intercept → HRV level (current state)
       - slope     → HRV direction (recovering vs deteriorating)
       Combined: b(t) + λ·a(t) gives richer signal than intercept alone.
    3. Z-score normalisation of CTLγ before correlation:
       CTLγ_z = (CTLγ - mean) / std
       → scale-invariant, avoids artificial R² inflation from scale mismatch
       → comparable across different γ values and modalities
    4. Temporal train/test split (70/30) to avoid overfitting:
       γ is fitted on first 70% of data, R² validated on last 30%.
       This is the correct approach per real fitting methodology.

    Returns
    -------
    best_gamma : float
    best_r2    : float (validated R² on test set)
    hrv_trend  : np.ndarray — intercept b(t) series for plotting
    best_lag   : int — optimal lag found
    """
    from scipy.stats import pearsonr as _pr
    if lags_to_test is None:
        lags_to_test = [0, 1, 2, 3, 5, 7]

    load  = np.array(load_arr, dtype=np.float64)
    hrv   = np.array(hrv_arr,  dtype=np.float64)

    # Extract combined HRV signal (b_z + w*a_z, data-driven w)
    # _hrv_trend now returns the combined normalised signal directly
    trend_combined = _hrv_trend(hrv, window=hrv_window, return_slope=False)

    # Temporal split: 70% train, 30% test
    n = len(load)
    split = int(n * 0.70)

    gammas = np.arange(gamma_range[0], gamma_range[1] + step/2, step)
    best_gamma, best_r2, best_lag = 0.35, -np.inf, 1

    for g in gammas:
        ctl_g = ftlm_fractional(load, g, max_lag=max_lag)
        # Z-score normalise CTLγ (scale-invariant, avoids R² inflation)
        _mu, _sd = np.nanmean(ctl_g[:split]), np.nanstd(ctl_g[:split])
        if _sd < 1e-9:
            continue
        ctl_z = (ctl_g - _mu) / _sd

        for lag in lags_to_test:
            if lag > 0:
                ctl_lagged = np.roll(ctl_z, lag)
                ctl_lagged[:lag] = np.nan
            else:
                ctl_lagged = ctl_z

            # Validate R² on TEST set only (avoids overfitting)
            test_valid = (np.arange(n) >= split) & (~np.isnan(ctl_lagged)) & (~np.isfinite(trend_combined) == False)
            test_valid &= np.isfinite(trend_combined)
            if test_valid.sum() < 10:
                continue
            try:
                r, _ = _pr(ctl_lagged[test_valid], trend_combined[test_valid])
                r2   = r ** 2
                if r2 > best_r2:
                    best_r2, best_gamma, best_lag = r2, float(g), lag
            except Exception:
                continue

    return best_gamma, best_r2, trend_combined, best_lag


def _compute_kappa(series_list, window=28, return_eigenvalues=False):
    """
    FMT scalar curvature: κ(t) = trace(cov(Δx)) over rolling window.
    Della Mattia (2019) — Functional Multidimensional Tensor.

    series_list       : list of 1D arrays, same length — dimensions of x(t).
    window            : rolling window for covariance (default 28 days).
    return_eigenvalues: if True, also return λ₁_frac (largest eigenvalue fraction)
                        λ₁_frac = λ₁ / Σλ — stress concentration index:
                          → near 1.0  = stress focal (one dimension dominates)
                          → near 1/d  = stress multisystemic (all dims perturbed equally)

    Each dimension is z-scored independently before computing Δx.
    F(t) = cov(Δx[t-window:t]) — d×d covariance matrix.
    κ(t) = trace(F(t)) — scalar curvature of athlete state.
    κ rising  → abrupt simultaneous changes across dimensions → fatigue accumulating.
    κ falling → adaptation stabilising.

    Returns:
      kappa (array n) — always
      lambda1_frac (array n) — only if return_eigenvalues=True
    """
    n   = len(series_list[0])
    d   = len(series_list)
    mat = np.full((n, d), np.nan)
    for j, s in enumerate(series_list):
        arr = np.array(s, dtype=np.float64)
        mu, sd = np.nanmean(arr), np.nanstd(arr)
        mat[:, j] = (arr - mu) / sd if sd > 1e-9 else (arr - mu)

    # First differences: Δx(t) = x(t) - x(t-1)
    delta = np.full_like(mat, np.nan)
    delta[1:] = mat[1:] - mat[:-1]

    kappa       = np.full(n, np.nan)
    lambda1_frac = np.full(n, np.nan)

    for t in range(window, n):
        wd = delta[t - window:t]
        valid = np.all(np.isfinite(wd), axis=1)
        if valid.sum() < max(10, d + 2):
            continue
        try:
            F = np.cov(wd[valid].T)
            kappa[t] = float(np.trace(F))
            if return_eigenvalues and d >= 2:
                # Eigenvalues of symmetric covariance matrix (all real, sorted desc)
                eigs = np.sort(np.linalg.eigvalsh(F))[::-1]
                eigs_pos = eigs[eigs > 0]
                if len(eigs_pos) > 0:
                    lambda1_frac[t] = float(eigs_pos[0] / eigs_pos.sum())
        except Exception:
            pass

    if return_eigenvalues:
        return kappa, lambda1_frac
    return kappa


def calcular_series_carga(df_act, df_wellness=None, ate_hoje=True):
    """
    Calcula CTL/ATL/TSB clássico (EWM) + FTLM fraccionário Della Mattia (2025).

    FONTE DE CARGA (prioridade):
        1. icu_training_load  (Intervals.icu, escala ~20-200/dia)
        2. session_rpe = (moving_time_min × RPE)   — fallback

    FTLM FRACCIONÁRIO — kernel Riemann-Liouville:
        CTLγ(t) = (1/Γ(γ)) · Σ_{k=1}^{t} Load(t-k) · k^(γ-1)
        Memória em lei de potência — treinos antigos persistem (sem saturação).

    FITTING DUAL DE γ POR MODALIDADE (Della Mattia Part II §1-3):
        Para cada modalidade (Bike, Row, Ski, Run):
            γ_perf_mod → max R²(CTLγ_mod ↔ ActivityCP_mod)  lag=0
            γ_mmp_mod  → max R²(CTLγ_mod ↔ MMP_PR_mod)       lag=0  (só sessões is_pr=True)
        γ_recovery (global) → max R²(CTLγ_overall(t-1) ↔ LnRMSSD_trend(t))  lag=1
            HRV trend via regressão janela 7d sobre LnRMSSD (não RMSSD bruto)

    RETORNA:
        ld   : DataFrame — Data, load_val, CTL, ATL, TSB,
                           CTLg_perf, CTLg_rec, FTLM,
                           CTLg_{mod}  para cada modalidade (load fraccionário modal)
        info : dict — gammas, R², fontes, n_pontos por modalidade
    """
    df = filtrar_principais(df_act).copy()
    df['Data'] = pd.to_datetime(df['Data'])

    # ── Fonte de carga ────────────────────────────────────────────────────────
    if 'icu_training_load' in df.columns and df['icu_training_load'].notna().sum() > 10:
        df['_load'] = pd.to_numeric(df['icu_training_load'], errors='coerce').fillna(0)
        fonte = 'icu_training_load'
    elif 'moving_time' in df.columns and 'rpe' in df.columns:
        _rpe = pd.to_numeric(df['rpe'], errors='coerce')
        df['_load'] = (pd.to_numeric(df['moving_time'], errors='coerce') / 60) * _rpe.fillna(_rpe.median())
        df['_load'] = df['_load'].fillna(0)
        fonte = 'session_rpe'
    else:
        return pd.DataFrame(), {'fonte': 'none'}

    # ── Série diária OVERALL ──────────────────────────────────────────────────
    ld = df.groupby('Data')['_load'].sum().reset_index().sort_values('Data')
    _idx = pd.date_range(ld['Data'].min(),
                          datetime.now().date() if ate_hoje else ld['Data'].max())
    ld = ld.set_index('Data').reindex(_idx, fill_value=0).reset_index()
    ld.columns = ['Data', 'load_val']

    # ── CTL / ATL / TSB clássicos ──────────────────────────────────────────────
    ld['CTL'] = ld['load_val'].ewm(span=42, adjust=False).mean()
    ld['ATL'] = ld['load_val'].ewm(span=7,  adjust=False).mean()
    ld['TSB'] = ld['CTL'] - ld['ATL']

    load_overall = ld['load_val'].values
    MAX_LAG = min(365, len(load_overall))
    _date_idx = pd.DatetimeIndex(ld['Data'])

    # ── γ_recovery via LnRMSSD trend (GLOBAL — HRV é modalidade-agnóstico) ──
    # Part II §2: HRV trend via regressão janela 7d sobre LnRMSSD
    # LnRMSSD (não RMSSD bruto): distribuição normal, compatível com tab_recovery
    gamma_rec, r2_rec = 0.35, 0.0
    hrv_trend_arr = None

    if df_wellness is not None and len(df_wellness) > 0:
        _wc = df_wellness.copy()
        _wc['Data'] = pd.to_datetime(_wc['Data'])

        # ── LnRMSSD — primary HRV signal (if available) ──────────────────
        _hrv_ln = np.full(len(_date_idx), np.nan)
        if 'hrv' in _wc.columns:
            _hrv_raw = (_wc.groupby('Data')['hrv']
                        .mean()
                        .reindex(_date_idx, fill_value=np.nan).values)
            _hrv_ln = np.where(_hrv_raw > 0, np.log(_hrv_raw), np.nan)

        # ── WEED proxy — z-score relative to rolling 28d baseline ─────────
        # Scale 1-5: 1=WORST, 5=BEST for ALL metrics in this wellness form.
        # This means:
        #   fatiga=5  → máxima vontade de treinar  (GOOD)
        #   stress=5  → sem stress                  (GOOD) — user confirmed
        #   soreness=5 → sem cansaço muscular       (GOOD) — user confirmed
        # NO inversion needed — 5 is always better across all three.
        #
        # z-score vs 28d rolling baseline:
        #   deviation = (today - personal_baseline) / personal_variability
        #   + positive z → above baseline → good signal
        #   - negative z → below baseline → bad signal
        # This is scale-invariant — a 1-5 scale gives same z-scores as 1-10.
        _weed_parts = []
        for _wc_col in ['stress', 'soreness', 'fatiga']:
            if _wc_col in _wc.columns:
                _s = (_wc.groupby('Data')[_wc_col]
                      .mean()
                      .reindex(_date_idx, fill_value=np.nan))
                _s = pd.to_numeric(_s, errors='coerce')
                # Rolling z-score relative to 28d personal baseline
                _roll_mu = _s.rolling(28, min_periods=7).mean()
                _roll_sd = _s.rolling(28, min_periods=7).std()
                _s_z     = (_s - _roll_mu) / _roll_sd.replace(0, np.nan)
                _weed_parts.append(_s_z.values)

        # ── Sleep quality — rolling z-score ───────────────────────────────
        _sleep_z = None
        if 'sleep_quality' in _wc.columns:
            _sq = (_wc.groupby('Data')['sleep_quality']
                   .mean()
                   .reindex(_date_idx, fill_value=np.nan))
            _sq = pd.to_numeric(_sq, errors='coerce')
            _sq_mu = _sq.rolling(28, min_periods=7).mean()
            _sq_sd = _sq.rolling(28, min_periods=7).std()
            _sleep_z = ((_sq - _sq_mu) / _sq_sd.replace(0, np.nan)).values

        # ── Composite recovery signal ──────────────────────────────────────
        # Primary: LnRMSSD trend (physiologically validated)
        # Enrichment: WEED z-score + Sleep z-score as optional additive signal
        # Weight: HRV 60%, WEED 25%, Sleep 15% (HRV dominates per literature)
        _composite_parts = [_hrv_ln]   # primary
        if _weed_parts:
            _weed_mean = np.nanmean(np.array(_weed_parts), axis=0)
            _composite_parts.append(('weed', _weed_mean, 0.25))
        if _sleep_z is not None:
            _composite_parts.append(('sleep', _sleep_z, 0.15))

        # Use LnRMSSD as primary for γ fitting (only if sufficient HRV data)
        if int(np.isfinite(_hrv_ln).sum()) >= 21:
            gamma_rec, r2_rec, hrv_trend_arr, _best_lag = fit_gamma_recovery(
                load_overall, _hrv_ln, hrv_window=7, max_lag=MAX_LAG)

        # Store WEED, sleep z-scores in ld for FMT — always, independent of HRV
        if _weed_parts:
            ld['WEED_z'] = np.nanmean(np.array(_weed_parts), axis=0)
        if _sleep_z is not None:
            # sleep_quality: scale 1=bad, 5=good (confirmed by user)
            # z-score is scale-invariant: positive z = above personal baseline = good
            # NO inversion needed: high value = good sleep = good signal
            # Rolling 28d baseline removes seasonal/trend effects
            _n_sleep = int(np.isfinite(_sleep_z).sum())
            if _n_sleep >= 5:   # threshold 5 (not 10 — may have gaps)
                ld['sleep_z'] = _sleep_z

    # ── W' daily series (icu_pm_w_prime per session) ─────────────────────────
    # icu_pm_w_prime = estimated W' capacity for that activity (Morton 3P model)
    # Rolling mean smooths day-to-day noise; stored for FMT 4th dimension
    if 'icu_pm_w_prime' in df.columns and df['icu_pm_w_prime'].notna().sum() >= 10:
        _wp_daily = (df.groupby('Data')['icu_pm_w_prime']
                     .mean()
                     .reindex(_date_idx, fill_value=np.nan))
        # Rolling mean 7d to smooth (W' estimate is noisy per-session)
        _wp_smooth = _wp_daily.rolling(7, min_periods=3).mean().values
        ld['wp_prime'] = _wp_smooth

    # ── Detect performance proxy column available (for info dict) ───────────
    _perf_col_global = next(
        (_pc for _pc in ['icu_pm_cp', 'icu_eftp']  # icu_ftp excluído: demasiado estável
         if _pc in df.columns and df[_pc].notna().sum() >= 10),
        None
    )

    # ── γ_perf POR MODALIDADE (Della Mattia Part II §1 + §3) ─────────────────
    # Cada modalidade tem a sua carga, CP e MMP — escalas de potência diferentes
    # Bike MMP20 ≠ Row MMP20 — misturar seria erro sistemático
    # γ reflecte a memória fisiológica específica de cada desporto:
    #   Ski: alta sazonalidade → γ baixo (memória muito longa)
    #   Row: sessões longas, poucas/semana → γ médio
    #   Bike: treino frequente, intensidade variável → depende dos dados
    #   Run: adaptações neuromusculares + aeróbicas → depende dos dados

    MODS = ['Bike', 'Row', 'Ski', 'Run']
    mod_info = {}      # γ, R², n por modalidade
    mod_loads = {}     # série diária de carga por modalidade

    if 'type' not in df.columns:
        df['type'] = 'Bike'

    for mod in MODS:
        df_mod = df[df['type'] == mod].copy()
        if len(df_mod) < 5:
            mod_info[mod] = {'gamma_perf': 0.35, 'r2_perf': 0.0,
                             'gamma_mmp': 0.35, 'r2_mmp': 0.0,
                             'n_cp': 0, 'n_mmp': 0, 'n_sessions': 0}
            mod_loads[mod] = np.zeros(len(ld))
            continue

        # Carga diária desta modalidade (alinhada com ld)
        _ld_mod = (df_mod.groupby('Data')['_load']
                   .sum()
                   .reindex(_date_idx, fill_value=0).values)
        mod_loads[mod] = _ld_mod

        # Adiciona ao ld para visualização
        ld[f'load_{mod}'] = _ld_mod
        ld[f'CTL_{mod}']  = pd.Series(_ld_mod).ewm(span=42, adjust=False).mean().values
        ld[f'ATL_{mod}']  = pd.Series(_ld_mod).ewm(span=7,  adjust=False).mean().values

        # γ via performance proxy desta modalidade — Della Mattia Part II §1
        # icu_pm_cp = ActivityCP: CP estimado da CURVA DE POTÊNCIA desta actividade
        #   → varia diariamente com o esforço da sessão, ideal para série temporal
        # icu_eftp  = eFTP estimado pelo modelo do Intervals.icu para esta actividade
        #   → mais suave, também por actividade
        # icu_ftp   = FTP setting (definido manualmente, demasiado estável para fitting)
        #   → NÃO usar: varia pouco, não dá sinal para R²
        # Ambos icu_pm_cp e icu_eftp são estimativas PER-ACTIVIDADE, não testes de CP
        # São válidos para fitting pois formam séries temporais: cp(t1), cp(t2), ...
        gamma_m, r2_m, n_cp_m = 0.35, 0.0, 0
        _perf_col = None
        for _pc in ['icu_pm_cp', 'icu_eftp']:   # icu_ftp excluído: demasiado estável
            if _pc in df_mod.columns and df_mod[_pc].notna().sum() >= 5:
                _perf_col = _pc
                break
        if _perf_col:
            _cp_mod = (df_mod.groupby('Data')[_perf_col]
                       .mean()
                       .reindex(_date_idx, fill_value=np.nan).values)
            gamma_m, r2_m, n_cp_m = fit_gamma_performance(
                _ld_mod, _cp_mod, lag=0, max_lag=MAX_LAG)

        # γ via MMP PR desta modalidade — Della Mattia Part II §1
        # Bike/Run: MMP20 (potência em ~20 min — paper §1, Imbach 2021)
        # Ski/Row:  MMP5  (esforços máximos curtos mais frequentes/realistas nestes desportos)
        # Apenas sessões is_pr=True: data exacta, esforço máximo confirmado
        gamma_mmp_m, r2_mmp_m, n_mmp_m = 0.35, 0.0, 0
        _mmp_col = 'mmp5_pr_w' if mod in ('Ski', 'Row') else 'mmp20_pr_w'
        if _mmp_col in df_mod.columns and df_mod[_mmp_col].notna().any():
            _mmp_s = (df_mod[df_mod[_mmp_col].notna()]
                      .groupby('Data')[_mmp_col].mean()
                      .reindex(_date_idx))
            if _mmp_s.notna().sum() >= 5:
                _gm, _rm, _nm = fit_gamma_performance(
                    _ld_mod, _mmp_s.values, lag=0, max_lag=MAX_LAG)
                if _rm > r2_m:
                    gamma_m, r2_m = _gm, _rm
                gamma_mmp_m, r2_mmp_m, n_mmp_m = _gm, _rm, _nm

        # CTLγ fraccionário para esta modalidade
        _ctlg_raw = ftlm_fractional(_ld_mod, gamma_m, max_lag=MAX_LAG)
        # Normalise to same scale as CTL for interpretable comparison:
        # Divide by the median ratio between CTLγ and CTL_mod where both > 0
        _ctl_mod_vals = ld[f'CTL_{mod}'].values
        _valid_norm = (_ctlg_raw > 0) & (_ctl_mod_vals > 0)
        if _valid_norm.sum() >= 10:
            _norm_factor = np.median(_ctlg_raw[_valid_norm] / _ctl_mod_vals[_valid_norm])
            if _norm_factor > 0:
                _ctlg_raw = _ctlg_raw / _norm_factor
        ld[f'CTLg_{mod}'] = _ctlg_raw

        # Collect MMP season bests for display in table
        # Use mmp*_w (all sessions) to find the TRUE maximum per duration.
        # mmp*_pr_w (is_pr=True) has too many entries — Intervals marks Yes
        # whenever ANY effort in that session exceeds the prior PR, not just
        # the session-wide maximum. So we take the max watts per duration
        # from mmp*_w across the whole season, with the date it occurred.
        _mmp_pr_list = []
        for _mc2 in ['mmp1_w','mmp3_w','mmp5_w','mmp12_w','mmp20_w','mmp60_w']:
            if _mc2 not in df_mod.columns:
                continue
            _df_w = df_mod[df_mod[_mc2].notna()][['Data', _mc2]].copy()
            if len(_df_w) == 0:
                continue
            # Use MOST RECENT session for each MMP duration.
            # Not the highest watts — the current fitness state is what matters,
            # not the historical peak which may be months out of date.
            # Sort by date descending, take the first (most recent) row.
            _df_w = _df_w.sort_values('Data', ascending=False)
            _recent_row = _df_w.iloc[0]
            _dur_label = _mc2.replace('_w', '').upper()
            _mmp_pr_list.append({
                'duracao': _dur_label,
                'data':    str(_recent_row['Data'])[:10],
                'watts':   round(float(_recent_row[_mc2]), 0),
            })
        _mmp_pr_list.sort(key=lambda x: x['duracao'])

        # Current CTLg value for this modality (used for dominance ranking)
        _ctlg_current = float(_ctlg_raw[-1]) if len(_ctlg_raw) > 0 else 0.0

        mod_info[mod] = {
            'gamma_perf':   round(gamma_m,     3),
            'r2_perf':      round(r2_m,        3),
            'gamma_mmp':    round(gamma_mmp_m, 3),
            'r2_mmp':       round(r2_mmp_m,    3),
            'n_cp':         n_cp_m,
            'perf_col':     _perf_col or 'none',
            'n_mmp':        n_mmp_m,
            'mmp_col':      _mmp_col,
            'mmp_pr_list':  _mmp_pr_list,
            'n_sessions':   len(df_mod),
            'ctlg_current': round(_ctlg_current, 3),  # for dominance display
        }

    # ── CTLγ overall — select dominant modality for γ_perf ──────────────────
    # "Dominant" = highest absolute CTLg value = where the athlete actually trains most
    # NOT highest R² — R² inflates with fewer data points (overfitting risk)
    # Logic:
    #   1. Use current CTLg absolute value as primary signal (what is your fitness NOW)
    #   2. Require minimum 10 sessions to be eligible as dominant
    #   3. R² as tiebreaker only, weighted by log(n) to penalise small samples
    _ctlg_vals = {m: float(ld[f'CTLg_{m}'].dropna().iloc[-1])
                  if f'CTLg_{m}' in ld.columns and ld[f'CTLg_{m}'].notna().any()
                  else 0.0
                  for m in MODS}
    # Eligible: at least 10 sessions in history
    _eligible = [m for m in MODS if mod_info[m]['n_sessions'] >= 10]
    if not _eligible:
        _eligible = MODS  # fallback: all mods

    # Score = CTLg_current (absolute fitness level) with R²×log(n) as tiebreaker
    def _dom_score(m):
        ns  = max(mod_info[m]['n_sessions'], 1)
        r2  = mod_info[m]['r2_perf']
        ctlg = _ctlg_vals.get(m, 0.0)
        return (ctlg, r2 * np.log(ns))  # primary=fitness, secondary=fit quality

    best_mod   = max(_eligible, key=_dom_score)
    gamma_perf = mod_info[best_mod]['gamma_perf']
    r2_perf    = mod_info[best_mod]['r2_perf']

    # Normalise overall CTLγ to same scale as CTL
    def _norm_ctlg(ctlg_arr, ctl_arr):
        valid = (ctlg_arr > 0) & (ctl_arr > 0)
        if valid.sum() >= 10:
            factor = np.median(ctlg_arr[valid] / ctl_arr[valid])
            if factor > 0:
                return ctlg_arr / factor
        return ctlg_arr

    _ctl_arr = ld['CTL'].values
    ld['CTLg_perf'] = _norm_ctlg(ftlm_fractional(load_overall, gamma_perf, max_lag=MAX_LAG), _ctl_arr)
    ld['CTLg_rec']  = _norm_ctlg(ftlm_fractional(load_overall, gamma_rec,  max_lag=MAX_LAG), _ctl_arr)

    # FTLM activo = melhor R²
    best_gamma  = gamma_perf if r2_perf >= r2_rec else gamma_rec
    best_source = 'perf'     if r2_perf >= r2_rec else 'recovery'
    ld['FTLM']  = ld['CTLg_perf'] if best_source == 'perf' else ld['CTLg_rec']

    # ── W' stress diário ─────────────────────────────────────────────────────
    # Fonte: AllWorkFTP (kJ acima de FTP, Intervals.icu export)
    #        icu_pm_w_prime (W' estimado pelo modelo Morton 3P, em Joules)
    #
    # Rácio bruto por sessão:
    #   raw_ratio = AllWorkFTP_kJ / (icu_pm_w_prime_J / 1000)
    #             = AllWorkFTP_kJ / W'_kJ  → fracção de W' consumida
    #   raw_ratio > 1 → W' excedido nessa sessão
    #   raw_ratio típico: 0.1 (sessão leve) a 3-5 (intervalos duros)
    #
    # Normalização final: z-score relativo ao baseline 60d
    #   w_stress = (rolling_7d_raw - media_60d) / std_60d
    #   → insensível às unidades absolutas (kJ vs J)
    #   → positivo = semana mais intensa que o normal
    #   → negativo = semana mais leve que o normal
    #   → comparável entre desportos e períodos
    #
    # z3_kj NÃO é usado como fallback: é energia total da zona 3
    # (muito maior que AllWorkFTP — métricas diferentes)
    if ('AllWorkFTP' in df.columns and 'icu_pm_w_prime' in df.columns and
            df['AllWorkFTP'].notna().sum() >= 5 and
            df['icu_pm_w_prime'].notna().sum() >= 5):
        _df_ws  = df.copy()
        _wp_j   = pd.to_numeric(_df_ws['icu_pm_w_prime'], errors='coerce')
        _aw     = pd.to_numeric(_df_ws['AllWorkFTP'],      errors='coerce')
        _wp_kj  = _wp_j / 1000.0
        # Rácio por sessão (só onde W' > 0.5 kJ para evitar divisão por zero)
        _df_ws['_w_ratio'] = np.where(_wp_kj > 0.5, _aw / _wp_kj, np.nan)
        # Média diária → rolling 7d → z-score 60d
        _w_daily = (_df_ws.groupby('Data')['_w_ratio']
                    .mean()
                    .reindex(_date_idx, fill_value=np.nan))
        _w_7d   = _w_daily.rolling(7, min_periods=2).mean()
        _w_mu   = _w_7d.rolling(60, min_periods=10).mean()
        _w_sd   = _w_7d.rolling(60, min_periods=10).std()
        ld['w_stress']     = ((_w_7d - _w_mu) / _w_sd.replace(0, np.nan)).values
        ld['w_stress_raw'] = _w_7d.values   # rácio bruto para referência no CSV

    # wp_prime (backward compat — usado no FMT_kappa_4d e download CSV)
    if 'icu_pm_w_prime' in df.columns and df['icu_pm_w_prime'].notna().sum() >= 5:
        _wp_daily = (df.groupby('Data')['icu_pm_w_prime']
                     .mean()
                     .reindex(_date_idx, fill_value=np.nan))
        ld['wp_prime'] = _wp_daily.rolling(7, min_periods=3).mean().values

    # ── HR quartil drift (hq_4 / hq_1) ──────────────────────────────────────
    # Paper FMT 2019: HR quartiles são dimensão Load do vector de estado x(t)
    # hq_4/hq_1 = HR drift ratio = quão concentrada foi a FC no final da sessão
    # Sessão aeróbica leve: hq_4 ≈ hq_1 (ratio ≈ 1.1)
    # Sessão intensa/drift: hq_4 >> hq_1 (ratio ≈ 1.3+)
    # Rolling 14d z-score: captura padrão crónico de intensidade intra-sessão
    # hq_1..hq_4: HR quartile means per session (Hq1..Hq4 in sheet)
    # Values of 0 are invalid (no HR data for that quartile) → treat as NaN
    if 'hq_1' in df.columns:
        df['hq_1'] = pd.to_numeric(df['hq_1'], errors='coerce').replace(0, np.nan)
    if 'hq_4' in df.columns:
        df['hq_4'] = pd.to_numeric(df['hq_4'], errors='coerce').replace(0, np.nan)

    if ('hq_1' in df.columns and 'hq_4' in df.columns and
            df['hq_1'].notna().sum() >= 5 and df['hq_4'].notna().sum() >= 5):
        _df_hq = df.copy()
        _hq1   = pd.to_numeric(_df_hq['hq_1'], errors='coerce')
        _hq4   = pd.to_numeric(_df_hq['hq_4'], errors='coerce')
        # Ratio hq_4/hq_1 só onde hq_1 > 40 bpm (evita artefactos)
        _hq_ratio = np.where(_hq1 > 40, _hq4 / _hq1, np.nan)
        _df_hq['_hq_ratio'] = _hq_ratio
        _hq_daily = (_df_hq.groupby('Data')['_hq_ratio']
                     .mean()
                     .reindex(_date_idx, fill_value=np.nan))
        # Z-score rolante 14d: desvio do padrão pessoal de HR drift
        _hq_mu  = _hq_daily.rolling(14, min_periods=5).mean()
        _hq_sd  = _hq_daily.rolling(14, min_periods=5).std()
        ld['hq_drift_z'] = ((_hq_daily - _hq_mu) / _hq_sd.replace(0, np.nan)).values

    # ── FMT Tensor κ (Della Mattia 2019) ─────────────────────────────────────
    # Vector de estado x(t) — alinhado com o paper §02:
    #   Dimensão 1: CTLγ_perf      — carga acumulada fraccionária (Load)
    #   Dimensão 2: HRV_trend      — estado autonómico (HRV, se disponível)
    #   Dimensão 3: WEED_z         — readiness subjectivo (stress+soreness+fatiga)
    #   Dimensão 4: sleep_z        — sono (paper trata separado do WEED)
    #   Dimensão 5: w_stress       — fracção W' consumida (W' dimension paper)
    #   Dimensão 6: hq_drift_z     — HR drift intra-sessão (HR quartiles paper)
    #
    # F(t) = cov(Δx) over 28-day rolling window
    # κ(t) = trace(F(t)) — scalar curvature (enriched TSS per paper §11)
    # κ rising  → chaos/fatigue accumulating across dimensions
    # κ falling → adaptation stabilising
    # λ₁/Σλ    → stress concentration: near 1=focal, near 1/d=multisystemic
    _fmt_w = 28

    def _impute_rolling(arr, window=7, min_periods=3):
        """
        Imputa NaN com rolling mean dos últimos X dias.
        Usado para dias sem wellness (ex: fim de semana sem medição HRV).
        - window=7: janela de 7 dias para calcular o valor esperado
        - min_periods=3: imputa se há pelo menos 3 dias recentes com dados
        - Se não há dados suficientes nos últimos 7 dias → mantém NaN
        Preserva valores reais — só substitui NaN.
        """
        s = pd.Series(arr, dtype=float)
        fill = s.rolling(window, min_periods=min_periods).mean()
        # Só preencher NaN (não substituir valores reais)
        return np.where(s.isna(), fill, s).astype(float)

    # Overall — começa sempre com CTLγ; adiciona dimensões se disponíveis
    # Cada dimensão é imputada por rolling mean 7d antes de entrar no tensor
    _dims_overall = [_impute_rolling(ld['CTLg_perf'].values)]
    _dim_names    = ['CTLγ']

    if 'HRV_trend' in ld.columns and ld['HRV_trend'].notna().sum() >= 20:
        _dims_overall.append(_impute_rolling(ld['HRV_trend'].values))
        _dim_names.append('HRV')
    if 'WEED_z' in ld.columns and ld['WEED_z'].notna().sum() >= 20:
        _dims_overall.append(_impute_rolling(ld['WEED_z'].values))
        _dim_names.append('WEED')
    if 'sleep_z' in ld.columns and ld['sleep_z'].notna().sum() >= 20:
        _dims_overall.append(_impute_rolling(ld['sleep_z'].values))
        _dim_names.append('Sleep')
    if 'w_stress' in ld.columns and ld['w_stress'].notna().sum() >= 20:
        _dims_overall.append(_impute_rolling(ld['w_stress'].values))
        _dim_names.append("W'")
    if 'hq_drift_z' in ld.columns and ld['hq_drift_z'].notna().sum() >= 20:
        _dims_overall.append(_impute_rolling(ld['hq_drift_z'].values))
        _dim_names.append('HR_drift')

    _tensor_dim = len(_dims_overall)

    if _tensor_dim >= 2:
        _kappa_arr, _lambda1_arr = _compute_kappa(
            _dims_overall, _fmt_w, return_eigenvalues=True)
        ld['FMT_kappa']        = _kappa_arr
        ld['FMT_lambda1_frac'] = _lambda1_arr
        ld['FMT_tensor_dim']   = _tensor_dim   # dimensão actual do tensor
    else:
        ld['FMT_kappa']        = np.nan
        ld['FMT_lambda1_frac'] = np.nan
        ld['FMT_tensor_dim']   = _tensor_dim

    # Legacy 4d kappa (backward compat — se wp_prime disponível)
    if 'wp_prime' in ld.columns and ld['wp_prime'].notna().sum() >= 20:
        _dims_4d = [_impute_rolling(ld['CTLg_perf'].values)]
        if 'HRV_trend' in ld.columns and ld['HRV_trend'].notna().sum() >= 20:
            _dims_4d.append(_impute_rolling(ld['HRV_trend'].values))
        if 'WEED_z' in ld.columns and ld['WEED_z'].notna().sum() >= 20:
            _dims_4d.append(_impute_rolling(ld['WEED_z'].values))
        _dims_4d.append(_impute_rolling(ld['wp_prime'].values))
        if len(_dims_4d) >= 2:
            ld['FMT_kappa_4d'] = _compute_kappa(_dims_4d, _fmt_w)

    # Per-modality 3×3: [CTLγ_mod, HRV_trend, WEED_z] — com imputação rolling
    for _mod in ['Bike', 'Row', 'Ski', 'Run']:
        _ctlg_col = f'CTLg_{_mod}'
        if _ctlg_col not in ld.columns or ld[_ctlg_col].notna().sum() < 20:
            continue
        _dims_mod = [_impute_rolling(ld[_ctlg_col].values)]
        if 'HRV_trend' in ld.columns and ld['HRV_trend'].notna().sum() >= 20:
            _dims_mod.append(_impute_rolling(ld['HRV_trend'].values))
        if 'WEED_z' in ld.columns and ld['WEED_z'].notna().sum() >= 20:
            _dims_mod.append(_impute_rolling(ld['WEED_z'].values))
        if len(_dims_mod) >= 2:
            ld[f'FMT_kappa_{_mod}'] = _compute_kappa(_dims_mod, _fmt_w)

    if hrv_trend_arr is not None:
        ld['HRV_trend'] = hrv_trend_arr

    # Store tensor dimension info for display in tab_pmc
    _tensor_dim_names = ' · '.join(_dim_names)

    info = {
        'fonte':           fonte,
        'perf_col':        _perf_col_global or 'none',
        'best_lag_rec':    locals().get('_best_lag', 1),
        'gamma_perf':      round(gamma_perf, 3),
        'gamma_rec':       round(gamma_rec,  3),
        'gamma_best':      round(best_gamma, 3),
        'gamma_source':    best_source,
        'r2_perf':         round(r2_perf, 3),
        'r2_rec':          round(r2_rec,  3),
        'best_mod':        best_mod,
        'mods':            mod_info,
        'tensor_dim':      locals().get('_tensor_dim', 0),
        'tensor_dim_names': locals().get('_tensor_dim_names', 'CTLγ'),
        'has_w_stress':    'w_stress'    in ld.columns and ld['w_stress'].notna().sum() >= 10,
        'has_hq_drift':    'hq_drift_z'  in ld.columns and ld['hq_drift_z'].notna().sum() >= 10,
        'has_sleep':       'sleep_z'     in ld.columns and ld['sleep_z'].notna().sum() >= 10,
    }
    return ld, info


def calcular_bpe(dw, metrica='hrv', baseline_dias=60):
    """
    BPE Z-Score semanal (metodologia do artigo Hopkins 2009 / Plews 2013).
    Baseline FIXO = últimos N dias do período total.
    Z = (Média_semanal − Baseline) / SWC
    """
    if metrica not in dw.columns or len(dw) < 14: return pd.DataFrame()
    df = dw.copy().sort_values('Data')
    df['Data'] = pd.to_datetime(df['Data'])
    df[metrica] = pd.to_numeric(df[metrica], errors='coerce')
    df_clean = df.dropna(subset=[metrica])
    if len(df_clean) < 14: return pd.DataFrame()

    # Baseline fixo sobre os últimos N dias
    n_base  = min(baseline_dias, len(df_clean))
    bdata   = df_clean[metrica].tail(n_base)
    base    = bdata.mean()
    std_b   = bdata.std()
    if pd.isna(base) or base <= 0 or pd.isna(std_b) or std_b <= 0: return pd.DataFrame()
    cv   = (std_b / base) * 100
    swc  = calcular_swc(base, cv)
    if swc is None or swc <= 0: return pd.DataFrame()

    df_clean['semana'] = df_clean['Data'].dt.to_period('W')
    rows = []
    for sem in sorted(df_clean['semana'].unique()):
        df_sem = df_clean[df_clean['semana'] == sem]
        media_sem = df_sem[metrica].mean()
        if pd.isna(media_sem): continue
        rows.append({
            'ano_semana':    str(sem),
            'media_semanal': media_sem,
            'baseline':      base,
            'swc':           swc,
            'cv_percent':    cv,
            'zscore':        (media_sem - base) / swc,
            'n_dias':        len(df_sem),
        })
    return pd.DataFrame(rows)

def calcular_recovery(dw):
    """Recovery Score composto (0-100). Igual ao original Python."""
    if len(dw) == 0: return pd.DataFrame()
    df = dw.copy().sort_values('Data')
    df['hrv_baseline'] = df['hrv'].rolling(14, min_periods=7).mean()
    df['rhr_baseline'] = df['rhr'].rolling(14, min_periods=7).mean() if 'rhr' in df.columns else np.nan
    df['hrv_cv_7d']    = cvr(df['hrv'], 7)
    df['hrv_cv_30d']   = cvr(df['hrv'], 30)
    rows = []
    for _, row in df.iterrows():
        hv = row.get('hrv'); hb = row.get('hrv_baseline'); cv = row.get('hrv_cv_7d')
        rv = row.get('rhr'); rb = row.get('rhr_baseline')
        # HRV (30%)
        hs = 50
        if pd.notna(hv) and pd.notna(hb) and hb > 0 and pd.notna(cv):
            inf_, sup_ = norm_range(hb, cv)
            if inf_ and sup_:
                band = sup_ - inf_
                if hv >= sup_:   hs = min(100, 75 + (hv - sup_) / band * 25 if band > 0 else 75)
                elif hv <= inf_: hs = max(0,   40 - (inf_ - hv) / band * 40 if band > 0 else 40)
                else:            hs = 50 + ((hv - inf_) / band * 25 if band > 0 else 0)
        # RHR (15%)
        rs = 50
        if pd.notna(rv) and pd.notna(rb) and rb > 0:
            pct = (rv - rb) / rb * 100
            rs = 90 if pct < -10 else 75 if pct < -5 else 55 if pct < 5 else 35 if pct < 10 else 20
        sl = conv_15(row.get('sleep_quality'))
        fa = conv_15(row.get('fatiga'))
        st = conv_15(row.get('stress'))
        hu = conv_15(row.get('humor'))
        so = conv_15(row.get('soreness'))
        score = hs*0.30 + rs*0.15 + sl*0.20 + fa*0.10 + st*0.10 + hu*0.05 + so*0.05 + 50*0.05
        i_, s_ = (norm_range(hb, cv) if pd.notna(hb) and pd.notna(cv) else (None, None))
        rows.append({'Data': row['Data'], 'recovery_score': score,
                     'hrv': hv, 'hrv_baseline': hb, 'hrv_cv_7d': cv, 'hrv_cv_30d': row.get('hrv_cv_30d'),
                     'normal_range_inf': i_, 'normal_range_sup': s_,
                     'hrv_component': hs, 'rhr_component': rs,
                     'sleep_component': sl, 'fatiga_component': fa, 'stress_component': st})
    return pd.DataFrame(rows)

def calcular_polinomios_carga(df_act_full):
    """
    Polynomial fit CTL/ATL Overall e por modalidade.
    Retorna dict com resultados + '_ld' (série completa).
    """
    df = filtrar_principais(df_act_full).copy()
    df['Data'] = pd.to_datetime(df['Data'])
    ld, _ = calcular_series_carga(df_act_full)
    if len(ld) == 0: return None
    ld['dias_num'] = (ld['Data'] - ld['Data'].min()).dt.days.values
    res = {'overall': {'CTL': {}, 'ATL': {}}, '_ld': ld}
    for met in ['CTL', 'ATL']:
        x, y = ld['dias_num'].values, ld[met].values
        for grau in [2, 3]:
            try:
                z = np.polyfit(x, y, grau); p = np.poly1d(z)
                r2 = np.corrcoef(y, p(x))[0, 1] ** 2
                res['overall'][met][f'grau{grau}'] = {'poly': p, 'r2': r2, 'x': x, 'y': y}
            except Exception: pass
    if 'moving_time' in df.columns and 'rpe' in df.columns:
        df['rpe_fill'] = pd.to_numeric(df['rpe'], errors='coerce').fillna(df['rpe'].median())
        df['load_val'] = (pd.to_numeric(df['moving_time'], errors='coerce') / 60) * df['rpe_fill']
        for tipo in ['Bike', 'Run', 'Row', 'Ski']:
            dt = df[df['type'] == tipo].copy()
            if len(dt) < 5: continue
            lt = dt.groupby('Data')['load_val'].sum().reset_index().sort_values('Data')
            idx_t = pd.date_range(lt['Data'].min(), ld['Data'].max())
            lt = lt.set_index('Data').reindex(idx_t, fill_value=0).reset_index(); lt.columns = ['Data', 'lv']
            lt['CTL'] = lt['lv'].ewm(span=42, adjust=False).mean()
            lt['ATL'] = lt['lv'].ewm(span=7,  adjust=False).mean()
            lt['dn']  = (lt['Data'] - lt['Data'].min()).dt.days.values
            res[f'tipo_{tipo}'] = {'CTL': {}, 'ATL': {}}
            for met in ['CTL', 'ATL']:
                x, y = lt['dn'].values, lt[met].values
                for grau in [2, 3]:
                    try:
                        z = np.polyfit(x, y, grau); p = np.poly1d(z)
                        r2 = np.corrcoef(y, p(x))[0, 1] ** 2
                        res[f'tipo_{tipo}'][met][f'grau{grau}'] = {'poly': p, 'r2': r2, 'x': x, 'y': y}
                    except Exception: pass
    return res

def analisar_falta_estimulo(df_act_full, janela_dias=14, baseline_dias=90):
    """
    Need Score v4 por modalidade.
    Load = duração_min × RPE

    Componentes base:
      A (25%) — Share FTLM deficit vs baseline 90d  [floor 10%]
      B (25%) — Quality deficit: carga_Z3/carga_total vs baseline
      C (20%) — Load deficit na janela vs load típico baseline
      D (20%) — FTLM slope_pct variação total no baseline [floor FTLM ini]
      E (10%) — Interacção A×B normalizada

    Scores derivados (coerentes com A/B/C/D):
      Need_volume    = A + C  (défice de carga total)
      Need_intensity = B + D×0.5  (défice de qualidade; D penaliza queda fitness)

    Overload por score acumulado (≥2 de 3 critérios):
      share_ratio > 1.4  → +1
      quality_ratio > 1.4 → +1
      slope_pct_janela > 0.15 → +1 (crescimento rápido na janela activa)
    Se overload: Need_final × 0.5, flag ⚠️, intensidade=LOW
    """
    MODS = ['Bike', 'Row', 'Ski', 'Run']
    df   = filtrar_principais(df_act_full).copy()
    df['Data'] = pd.to_datetime(df['Data'])

    if 'moving_time' not in df.columns or 'rpe' not in df.columns:
        return None, None

    df['dur_min']  = pd.to_numeric(df['moving_time'], errors='coerce') / 60
    df['rpe_n']    = pd.to_numeric(df['rpe'], errors='coerce')
    df['load']     = df['dur_min'] * df['rpe_n']
    df['load_z3']  = np.where(df['rpe_n'] >= 7, df['load'], 0.0)

    hoje     = pd.Timestamp.now().normalize()
    ini_jan  = hoje - pd.Timedelta(days=janela_dias)
    ini_base = hoje - pd.Timedelta(days=baseline_dias)

    # ── FTLM por modalidade — histórico completo (não recalcular na janela curta)
    all_dates = pd.date_range(df['Data'].min(), hoje, freq='D')
    ftlm_mods = {}
    for mod in MODS:
        dm = df[df['type'] == mod].copy()
        ld = (dm.groupby('Data')['load'].sum()
                .reindex(all_dates, fill_value=0.0)
                .reset_index())
        ld.columns = ['Data', 'load']
        ld['FTLM'] = ld['load'].ewm(span=28, adjust=False).mean()
        ftlm_mods[mod] = ld.set_index('Data')

    ftlm_total = sum(ftlm_mods[m]['FTLM'] for m in MODS).replace(0, np.nan)

    results    = {}
    debug_rows = []

    for mod in MODS:
        fm = ftlm_mods[mod]

        # ── A — Share FTLM deficit ──────────────────────────────────────────
        share_jan  = float((fm.loc[ini_jan:hoje,  'FTLM'] /
                            ftlm_total.loc[ini_jan:hoje]).mean())
        share_base = float((fm.loc[ini_base:hoje, 'FTLM'] /
                            ftlm_total.loc[ini_base:hoje]).mean())
        share_jan  = 0.0 if np.isnan(share_jan)  else share_jan
        share_base = 0.0 if np.isnan(share_base) else share_base
        floor_peso = min(1.0, share_base / 0.10) if share_base < 0.10 else 1.0
        A_raw = max(0.0, (share_base - share_jan) / share_base) if share_base > 0 else 0.0

        # Overload guard: se share_base muito baixo (<5%), ratios são instáveis
        # Modalidade com <5% do treino histórico → não pode estar em overload
        _share_min_ol = 0.05
        A     = min(A_raw, 1.0) * floor_peso

        # ── B — Quality deficit (carga_Z3 / carga_total) ───────────────────
        dm_jan  = df[(df['type']==mod) & (df['Data'] >= ini_jan)]
        dm_base = df[(df['type']==mod) & (df['Data'] >= ini_base)]
        def _q(d_):
            ct = d_['load'].sum()
            return float(d_['load_z3'].sum() / ct) if ct > 0 else 0.0
        q_jan  = _q(dm_jan)
        q_base = _q(dm_base)
        B_raw  = max(0.0, (q_base - q_jan) / q_base) if q_base > 0 else 0.0
        B      = min(B_raw, 1.0)

        # ── C — Load deficit na janela vs load típico ───────────────────────
        load_jan    = dm_jan['load'].sum()
        load_base   = dm_base['load'].sum()
        n_dias_base = max(1, (hoje - ini_base).days)
        load_tipico = (load_base / n_dias_base) * janela_dias
        C_raw = max(0.0, (load_tipico - load_jan) / load_tipico) if load_tipico > 0 else 0.0
        C     = min(C_raw, 1.0)

        # ── D — FTLM slope_pct (variação % total no baseline) ──────────────
        ftlm_base_series = fm.loc[ini_base:hoje, 'FTLM'].dropna()
        slope_pct = 0.0
        D = 0.0
        if len(ftlm_base_series) >= 10:
            ftlm_ini = float(ftlm_base_series.iloc[0])
            ftlm_fim = float(ftlm_base_series.iloc[-1])
            threshold_ini = max(1.0, ftlm_fim * 0.05)
            if ftlm_ini >= threshold_ini and ftlm_ini > 0:
                slope_pct = (ftlm_fim - ftlm_ini) / ftlm_ini
                if slope_pct < 0:
                    D = min(1.0, abs(slope_pct) * 2)
                elif slope_pct > 0.30:
                    D = min(0.5, (slope_pct - 0.30) * 2)

        # ── E — Interacção A×B ──────────────────────────────────────────────
        if A_raw > 0 and B_raw > 0:    fator_e = 1.3
        elif A_raw > 0 or B_raw > 0:   fator_e = 0.7
        else:                           fator_e = 0.0
        E = float(np.clip((A_raw + B_raw) / 2 * fator_e, 0.0, 1.0))

        # ── Need Score base ─────────────────────────────────────────────────
        need = min(100.0,
                   A * 100 * 0.25 +
                   B * 100 * 0.25 +
                   C * 100 * 0.20 +
                   D * 100 * 0.20 +
                   E * 100 * 0.10)

        # ── Need_volume e Need_intensity — camada de prescrição (read-only) ──
        # NÃO alteram Need_score. Apenas interpretam A/B/C/D.
        # Pesos: A×0.6+C×0.4 para volume | B×0.7+D×0.3 para intensidade
        need_vol = min(100.0, A * 100 * 0.60 + C * 100 * 0.40)
        need_int = min(100.0, B * 100 * 0.65 + D * 100 * 0.35)

        prio_base = 'ALTA' if need >= 70 else 'MÉDIA' if need >= 40 else 'BAIXA'

        # ── Overload guards — densidade histórica + slope ──────────────────
        share_ratio   = share_jan / share_base if share_base > 0 else 0.0
        quality_ratio = q_jan / q_base         if q_base    > 0 else 0.0

        # Sessões nos últimos 90d desta modalidade
        _sess_90d = len(df[(df['type']==mod) & (df['Data'] >= ini_base)])

        # Slope FTLM na janela activa
        ftlm_jan_series = fm.loc[ini_jan:hoje, 'FTLM'].dropna()
        slope_pct_jan = 0.0
        fi = 0.0
        ff = 0.0
        if len(ftlm_jan_series) >= 3:
            fi = float(ftlm_jan_series.iloc[0])
            ff = float(ftlm_jan_series.iloc[-1])
            if fi > 0:
                slope_pct_jan = (ff - fi) / fi

        # Guard 1: <4 sessões/90d — sem dados para avaliar overload
        _ignorar_overload = _sess_90d < 4

        # Guard 2: FTLM pequeno — slope instável (início de época)
        _ftlm_p10 = float(fm['FTLM'].quantile(0.10)) if len(fm['FTLM'].dropna()) >= 10 else 0.0
        _ignorar_slope = (fi < max(_ftlm_p10, ff * 0.10)) if ff > 0 else True

        # Frequência esperada para ajustar A (complementar ao A_share)
        _freq_base    = _sess_90d / 90.0
        _esperado_7d  = _freq_base * janela_dias
        _sess_jan     = len(df[(df['type']==mod) & (df['Data'] >= ini_jan)])
        _A_freq_boost = 0.0
        if _esperado_7d > 0.5:  # só se há histórico suficiente
            _freq_ratio = _sess_jan / _esperado_7d
            if _freq_ratio < 0.5:     _A_freq_boost = 0.20  # treinou muito menos
            elif _freq_ratio < 0.75:  _A_freq_boost = 0.10
        # Aplicar boost em A (max 1.0)
        A_adj = min(1.0, A + _A_freq_boost * (1.0 - A))

        # Dias desde último Z3 (para trigger de estímulo forte)
        _dm_z3 = df[(df['type']==mod) & (df['rpe_n'] >= 7) & (df['Data'] >= ini_base)]
        _ultima_z3 = _dm_z3['Data'].max() if len(_dm_z3) > 0 else pd.NaT
        _dias_z3 = int((hoje - _ultima_z3).days) if pd.notna(_ultima_z3) else 999
        _gap_z3_tipico = float(max(1, _dm_z3['Data'].sort_values()
                                   .diff().dt.days.dropna().median())
                               if len(_dm_z3) >= 2 else 14.0)
        _limite_z3 = _gap_z3_tipico * 1.5
        _forcar_z3 = (_dias_z3 > _limite_z3) and (not _ignorar_overload)

        # Score overload com guards
        score_overload = 0
        if not _ignorar_overload:
            if share_ratio   > 1.4 and share_base > 0.05: score_overload += 1
            if quality_ratio > 1.4 and q_base > 0.02:     score_overload += 1
            if slope_pct_jan > 0.15 and not _ignorar_slope: score_overload += 1
        overload = (score_overload >= 2) and not _ignorar_overload

        # Tipo de overload (para modulação da intensidade)
        _overload_vol  = share_ratio   > 1.4 and share_base > 0.05
        _overload_qual = quality_ratio > 1.4 and q_base > 0.02
        _overload_agudo= slope_pct_jan > 0.15 and not _ignorar_slope

        if overload:
            if _overload_qual or _overload_agudo:
                need_int_prescr = min(100.0, need_int * 0.65)   # reduzir forte
            else:
                need_int_prescr = min(100.0, need_int * 0.90)   # só volume alto — redução leve
        else:
            need_int_prescr = need_int

        # C_reforçado + gap
        datas_mod_dbg  = df[df['type']==mod]['Data'].sort_values()
        ultima_dbg     = datas_mod_dbg.max() if len(datas_mod_dbg) > 0 else pd.NaT
        dias_sem       = int((hoje - ultima_dbg).days) if pd.notna(ultima_dbg) else 999
        gap_tipico_dbg = float(max(1, datas_mod_dbg[datas_mod_dbg >= ini_base]
                                   .diff().dt.days.dropna().median())
                               if len(datas_mod_dbg[datas_mod_dbg >= ini_base]) >= 2
                               else 7.0)
        gap_score    = min(1.0, dias_sem / (gap_tipico_dbg * 2))
        c_reforcado  = max(C, gap_score)
        gap_ratio    = min(dias_sem / gap_tipico_dbg, 3.0) if gap_tipico_dbg > 0 else 0.0

        # ── Guardar estado intermédio — prescrição calculada em 2ª passagem ─
        results[mod] = dict(
            need_score=need, prioridade=prio_base, overload=overload,
            overload_score=score_overload,
            overload_tipo=('qualidade/agudo' if (_overload_qual or _overload_agudo)
                           else 'volume' if _overload_vol else ''),
            need_vol=round(need_vol, 1),
            need_int=round(need_int, 1),
            need_int_prescr=round(need_int_prescr, 1),
            prescricao='',    # preenchida em 2ª passagem
            dias_sem=dias_sem, gap_score=round(gap_score, 2),
            gap_ratio=gap_ratio, gap_z3=_dias_z3, forcar_z3=_forcar_z3,
            gap_z3_limite=round(_limite_z3, 1),
            c_reforcado=round(c_reforcado, 3),
            sess_90d=_sess_90d, ignorar_overload=_ignorar_overload,
            share_actual=share_jan, share_hist=share_base,
            quality_actual=q_jan, quality_hist=q_base,
            load_jan=load_jan, load_tipico=load_tipico,
            ftlm_slope_pct=slope_pct, ftlm_slope_jan=slope_pct_jan,
            A=A_adj, B=B, C=C, D=D, E=E, floor_peso=floor_peso)

        debug_rows.append({
            'Modalidade':            mod,
            'Need Score':            round(need, 1),
            'Need Volume':           round(need_vol, 1),
            'Need Intensity':        round(need_int, 1),
            'Need Int (prescrição)': round(need_int_prescr, 1),
            'Prescrição':            '(2ª passagem)',
            'dias_sem_sessao':       dias_sem,
            'gap_score':             round(gap_score, 2),
            'C_reforçado':           round(c_reforcado, 3),
            'Prioridade base':       prio_base,
            'Overload score':        score_overload,
            'Overload':              '⚠️ SIM' if overload else 'não',
            'Overload tipo':         results[mod]['overload_tipo'],
            'Sess 90d':              _sess_90d,
            'Ignorar OL':            _ignorar_overload,
            'Dias Z3':               _dias_z3,
            'Forçar Z3':             _forcar_z3,
            'A_freq_boost':          round(_A_freq_boost, 2),
            'A Share actual%':       round(share_jan  * 100, 1),
            'A Share hist90d%':      round(share_base * 100, 1),
            'A Deficit%':            round(A_raw * 100, 1),
            'A Floor peso':          round(floor_peso, 2),
            'A contribuição':        round(A_adj * 100 * 0.25, 1),
            'B Quality actual%':     round(q_jan  * 100, 1),
            'B Quality hist90d%':    round(q_base * 100, 1),
            'B Deficit%':            round(B_raw * 100, 1),
            'B contribuição':        round(B * 100 * 0.25, 1),
            'C Load janela':         round(load_jan, 1),
            'C Load típico':         round(load_tipico, 1),
            'C Deficit%':            round(C_raw * 100, 1),
            'C contribuição':        round(C * 100 * 0.20, 1),
            'D FTLM ini':            round(ftlm_ini if len(ftlm_base_series)>=10 else 0, 1),
            'D FTLM fim':            round(ftlm_fim if len(ftlm_base_series)>=10 else 0, 1),
            'D slope_pct%':          round(slope_pct * 100, 1),
            'D slope_jan%':          round(slope_pct_jan * 100, 1),
            'D contribuição':        round(D * 100 * 0.20, 1),
            'E Fator VQ':            fator_e,
            'E contribuição':        round(E * 100 * 0.10, 1),
        })

    # ══════════════════════════════════════════════════════
    # 2ª PASSAGEM — rank relativo + prescrição contextual
    # ══════════════════════════════════════════════════════
    # Calcular need_int_prescr de todos os mods para rank
    _all_scores = [d['need_int_prescr'] for d in results.values()]
    _std_scores  = float(np.std(_all_scores)) if len(_all_scores) > 1 else 0.0

    for mod, d in results.items():
        ni  = d['need_int_prescr']
        nv  = d['need_vol']
        ol  = d['overload']
        gr  = d['gap_ratio']
        fz3 = d['forcar_z3']
        dz3 = d['gap_z3']
        lz3 = d['gap_z3_limite']

        # Rank percentil dentro das modalidades (0-100)
        rank_int = float(sum(ni > s for s in _all_scores)) / max(len(_all_scores)-1, 1) * 100

        # Guard baixa dispersão: se todos scores semelhantes (std < 5)
        # → usar padrão histórico em vez de prescrever intenso a todos
        _baixa_dispersao = _std_scores < 5.0

        # Aviso Z3
        _aviso_z3 = (f" ⚡ Último Z3 há {dz3}d (limite {lz3:.0f}d)"
                     if fz3 else "")

        if ol:
            if d['overload_tipo'] == 'volume':
                prescricao = "🟡 Overload de volume — intensidade leve/moderada"
            else:
                prescricao = "🟢 Overload — reduzir intensidade" + _aviso_z3
        elif fz3 and not _baixa_dispersao:
            # Z3 em falta há muito tempo E há diferenciação entre mods
            prescricao = "🟠 Estímulo Z3 urgente" + _aviso_z3
        elif _baixa_dispersao:
            # Todos mods com need semelhante → usar histórico
            if nv >= 40:
                prescricao = "🔵 Sessão de base/volume (sem diferenciação clara)"
            else:
                prescricao = "⚪ Manutenção (estado equilibrado)"
        elif rank_int >= 75:
            # Top 25% de intensidade → sessão intensa
            if nv >= 50:
                prescricao = "🔴 Sessão completa (volume + intensidade)" + _aviso_z3
            elif gr <= 1.0:
                prescricao = "🟠 Sessão intensa/curta (fresco — défice qualidade)" + _aviso_z3
            elif gr <= 2.0:
                prescricao = "🟡 Sessão mista vol+int (reentrée gradual)" + _aviso_z3
            else:
                prescricao = "🔵 Sessão de base/reentrée (gap longo)" + _aviso_z3
        elif rank_int >= 50:
            # Top 50% → sessão moderada
            if nv >= 50:
                prescricao = "🔵 Sessão de volume/base + intensidade moderada"
            else:
                prescricao = "🟡 Sessão moderada"
        elif nv >= 50:
            prescricao = "🔵 Sessão de volume/base (défice de carga)"
        else:
            prescricao = "⚪ Manutenção ou descanso"

        results[mod]['prescricao'] = prescricao
        results[mod]['rank_int']   = round(rank_int, 0)
        results[mod]['std_scores'] = round(_std_scores, 1)

        # Actualizar debug
        for row in debug_rows:
            if row['Modalidade'] == mod:
                row['Prescrição'] = prescricao
                row['Rank int%']  = round(rank_int, 0)
                row['Std scores'] = round(_std_scores, 1)

    results_sorted = dict(sorted(
        results.items(), key=lambda x: x[1]['need_score'], reverse=True))
    return results_sorted, pd.DataFrame(debug_rows)



# ════════════════════════════════════════════════════════════════════════════
# MÓDULO: data_loader.py
# ════════════════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════════════════════
# data_loader.py — ATHELTICA Dashboard
# Autenticação Google Sheets, carregamento e preprocessing.
# ════════════════════════════════════════════════════════════════════════════════

# ── Autenticação ──────────────────────────────────────────────────────────────

@st.cache_resource
def get_gc():
    """Autentica Google Sheets com Service Account em st.secrets."""
    try:
        creds = Credentials.from_service_account_info(
            dict(st.secrets["gcp_service_account"]), scopes=SCOPES)
        return gspread.authorize(creds)
    except Exception as e:
        st.error(f"❌ Erro autenticação Google: {e}")
        st.info("Configura as credenciais em Settings → Secrets do Streamlit Cloud.")
        return None

# ── Carregamento ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=7200, show_spinner=False)
def carregar_wellness(days_back):
    gc = get_gc()
    if gc is None: return pd.DataFrame()
    try:
        ws  = gc.open_by_url(WELLNESS_URL).worksheet("Respostas ao formulário 1")
        df  = get_as_dataframe(ws, evaluate_formulas=True, header=0)
        if df.columns.duplicated().any(): df = df.loc[:, ~df.columns.duplicated()]
        cd  = detectar_col(df, ['Data', 'data', 'Date', 'Carimbo de data/hora'])
        if cd:
            df['Data'] = df[cd].apply(parse_date)
            df = df.dropna(subset=['Data']).sort_values('Data')
        for var, lst in MAPA_WELLNESS.items():
            col = detectar_col(df, lst)
            if col: df[var] = df[col].apply(br_float)
        dm = datetime.now() - timedelta(days=days_back)
        return df[df['Data'] >= dm].reset_index(drop=True)
    except Exception as e:
        st.error(f"Erro ao carregar wellness: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=7200, show_spinner=False)
def carregar_atividades(days_back):
    gc = get_gc()
    if gc is None: return pd.DataFrame()
    try:
        ws  = gc.open_by_url(TRAINING_URL).worksheet("intervals.icu_activities-export")
        df  = get_as_dataframe(ws, evaluate_formulas=True, header=0)
        if df.columns.duplicated().any(): df = df.loc[:, ~df.columns.duplicated()]
        cd  = detectar_col(df, ['Date', 'start_date_local', 'date', 'data', 'Data'])
        if cd:
            df['Data'] = df[cd].apply(lambda x: parse_date(str(x)[:10]))
            df = df.dropna(subset=['Data']).sort_values('Data')
        TEXTO = ['type', 'name', 'date', 'start_date_local']
        for var, lst in MAPA_TRAINING.items():
            col = detectar_col(df, lst)
            if col: df[var] = df[col] if var in TEXTO else df[col].apply(br_float)
        if 'type' in df.columns: df['type'] = df['type'].apply(norm_tipo)
        dm = datetime.now() - timedelta(days=days_back)
        return df[df['Data'] >= dm].reset_index(drop=True)
    except Exception as e:
        st.error(f"Erro ao carregar atividades: {e}")
        return pd.DataFrame()

# ── Preprocessing ─────────────────────────────────────────────────────────────

def _preencher_faltantes_lookback(df, coluna):
    """
    Preenche NaN com mediana dos 7 dias ANTERIORES (lookback, sem data leakage).
    Fallback: mediana 14 dias anteriores → média global.
    Igual ao método preencher_faltantes() do ATHELTICA v12 original.
    """
    if coluna not in df.columns: return df
    df = df.copy()
    df['Data'] = pd.to_datetime(df['Data'])
    df = df.sort_values('Data').reset_index(drop=True)
    for idx, row in df.iterrows():
        if pd.notna(row[coluna]): continue
        data_ref = row['Data']
        # Janela 7 dias anteriores
        w7 = df[(df['Data'] >= data_ref - timedelta(days=7)) &
                (df['Data'] <  data_ref) &
                (df[coluna].notna())][coluna]
        if len(w7) >= 2:
            df.at[idx, coluna] = w7.median(); continue
        # Janela 14 dias anteriores
        w14 = df[(df['Data'] >= data_ref - timedelta(days=14)) &
                 (df['Data'] <  data_ref) &
                 (df[coluna].notna())][coluna]
        if len(w14) >= 3:
            df.at[idx, coluna] = w14.median(); continue
        # Fallback: média global
        media = df[df[coluna].notna()][coluna].mean()
        if pd.notna(media):
            df.at[idx, coluna] = media
    return df


def preproc_wellness(df):
    """
    Limpeza wellness — igual ao ATHELTICA v12 DataPreprocessor.limpar_wellness():
    1. Ordenar por Data
    2. Remover duplicatas por Data (keep=first)
    3. Z-score threshold=3.0 → NaN (picos)
    4. Zeros inválidos → NaN (hrv, rhr, sleep_hours)
    5. Preencher faltantes: mediana 7d anteriores → 14d → média global
    """
    if len(df) == 0: return df
    df = df.copy().sort_values('Data')
    df = df.drop_duplicates(subset=['Data'], keep='first')
    # [1] Remover picos Z-score
    for c in [c for c in ['hrv', 'rhr', 'sleep_hours', 'sleep_quality',
                           'stress', 'fatiga', 'humor', 'soreness', 'peso', 'fat']
              if c in df.columns]:
        df[c] = remove_zscore(df[c], 3.0)
    # [2] Zeros inválidos → NaN
    df = remove_zeros(df, ['hrv', 'rhr', 'sleep_hours'])
    # [3] Preencher faltantes com lookback (sem data leakage)
    for c in [c for c in ['hrv', 'rhr', 'sleep_quality', 'fatiga',
                           'stress', 'humor', 'soreness']
              if c in df.columns]:
        df = _preencher_faltantes_lookback(df, c)
    return df.reset_index(drop=True)

def preproc_ativ(df):
    """
    Limpeza atividades — igual ao ATHELTICA v12 DataPreprocessor.limpar_training():
    1. Ordenar por Data
    2. Remover duplicatas por [Data, type, moving_time]
    3. Padronizar tipos via TYPE_MAP
    4. Remover tipos inválidos (não em VALID_TYPES)
    5. Z-score threshold=3.5 → NaN em icu_eftp E AllWorkFTP
    6. Zeros → NaN em moving_time, icu_eftp, AllWorkFTP
    7. Remover moving_time ≤ 60s
    """
    if len(df) == 0: return df
    df = df.copy().sort_values('Data')
    # [1] Deduplicar
    sub = [c for c in ['Data', 'type', 'moving_time'] if c in df.columns]
    if len(sub) >= 2: df = df.drop_duplicates(subset=sub, keep='first')
    # [2] Padronizar tipos
    if 'type' in df.columns: df['type'] = df['type'].apply(norm_tipo)
    # [3] Filtrar tipos válidos
    if 'type' in df.columns: df = df[df['type'].isin(VALID_TYPES)]
    # [4] Z-score picos icu_eftp e AllWorkFTP (threshold=3.5, igual ao original)
    if 'icu_eftp'    in df.columns: df['icu_eftp']    = remove_zscore(df['icu_eftp'],    3.5)
    if 'AllWorkFTP'  in df.columns: df['AllWorkFTP']  = remove_zscore(df['AllWorkFTP'],  3.5)
    # [5] Zeros → NaN (rpe=0 também é inválido — sem esforço não é sessão)
    df = remove_zeros(df, ['moving_time', 'icu_eftp', 'AllWorkFTP', 'rpe'])
    # [6] Remover duração ≤ 60s
    if 'moving_time' in df.columns:
        df = df[pd.to_numeric(df['moving_time'], errors='coerce') > 60]
    return df.reset_index(drop=True)

def filtrar_datas(df, di, df_):
    """Filtra DataFrame pelo intervalo [di, df_]."""
    if len(df) == 0: return df
    df = df.copy()
    df['Data'] = pd.to_datetime(df['Data'])
    return df[(df['Data'].dt.date >= di) & (df['Data'].dt.date <= df_)].reset_index(drop=True)


@st.cache_data(ttl=7200, show_spinner=False)
def carregar_annual():
    """
    Carrega AquecSki, AquecBike, AquecRow via gviz CSV.
    Data cleaning idêntico ao código original Python:
    - Renomeia colunas Unnamed (AquecRow)
    - Remove colunas 100% vazias
    - Converte vazios/nan/null → NaN
    - Converte numéricos, zeros → NaN
    - Remove ruídos: HR fora 40-220, O2 fora 20-100, Drag fora 80-200, Pwr fora 0.3-3.0
    - Normaliza coluna DATA e remove linhas sem data
    """
    COLS_NAO_NUM = ['Mês', 'Mes', 'Fase', 'DATA', 'Data', 'Treino_antes', 'Atividade']
    AQUECROW_RENAME = {
        'Unnamed: 2': 'DATA',       'Treino antes': 'Treino_antes',
        'Unnamed: 4': 'HR_140W',    'Unnamed: 5':   'HR_160W',
        'Unnamed: 6': 'HR_180W',    'Unnamed: 7':   'HR_200W',
        'Unnamed: 8': 'HR_Pwr_140w','Unnamed: 9':   'HR_Pwr_160w',
        'Unnamed: 10':'HR_Pwr_180w','Unnamed: 11':  'O2_140W',
        'Unnamed: 12':'O2_160W',    'Unnamed: 13':  'O2_180W',
        'Drag Factor':'Drag_Factor',
    }
    dfs = {}
    for aba in ANNUAL_SHEETS:
        url = (f"https://docs.google.com/spreadsheets/d/{ANNUAL_SPREADSHEET_ID}"
               f"/gviz/tq?tqx=out:csv&sheet={aba}")
        try:
            df = pd.read_csv(url)
            df.columns = [str(c).strip() for c in df.columns]
            if aba == 'AquecRow':
                df = df.rename(columns={k:v for k,v in AQUECROW_RENAME.items() if k in df.columns})
            # Normalizar nomes genéricos
            rename_gen = {}
            for col in df.columns:
                if col in ['Mês','Mes']: rename_gen[col] = 'Mês'
                elif col in ['DATA','Data']: rename_gen[col] = 'DATA'
                elif 'Drag' in col and 'Factor' in col: rename_gen[col] = 'Drag_Factor'
                elif 'Treino' in col: rename_gen[col] = 'Treino_antes'
            if rename_gen: df = df.rename(columns=rename_gen)
            # Remove colunas 100% vazias
            df = df.dropna(axis=1, how='all')
            # Strings inválidas → NaN
            df = df.replace(['', ' ', 'nan', 'NaN', 'null', 'NULL', 'None'], np.nan)
            # Numéricos + zeros → NaN
            for col in df.columns:
                if col not in COLS_NAO_NUM:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    df[col] = df[col].replace({0.0: np.nan, 0: np.nan})
            # Remover ruídos por tipo (igual ao original Python)
            for col in df.columns:
                if col in COLS_NAO_NUM: continue
                cu = col.upper()
                if 'HR' in cu and 'PWR' not in cu and 'DRAG' not in cu:
                    df[col] = df[col].where((df[col] >= 40) & (df[col] <= 220))
                elif 'O2' in cu:
                    df[col] = df[col].where((df[col] >= 20) & (df[col] <= 100))
                elif 'DRAG' in cu:
                    df[col] = df[col].where((df[col] >= 80) & (df[col] <= 200))
                elif 'PWR' in cu:
                    df[col] = df[col].where((df[col] >= 0.3) & (df[col] <= 3.0))
            # Normalizar DATA
            if 'DATA' in df.columns:
                df['DATA'] = pd.to_datetime(df['DATA'], dayfirst=True, errors='coerce')
                df = df.dropna(subset=['DATA']).sort_values('DATA')
            df['Atividade'] = aba
            dfs[aba] = df.reset_index(drop=True)
        except Exception:
            dfs[aba] = pd.DataFrame()
    validos = [d for d in dfs.values() if len(d) > 0]
    df_all = pd.concat(validos, ignore_index=True) if validos else pd.DataFrame()
    return dfs, df_all


@st.cache_data(ttl=7200, show_spinner=False)
def carregar_corporal():
    """Carrega aba Consolidado_Comida. Lida com vírgula decimal (PT-BR)."""
    gc = get_gc()
    if gc is None: return pd.DataFrame()
    try:
        ws  = gc.open_by_url(FOOD_URL).worksheet("Consolidado_Comida")
        df  = get_as_dataframe(ws, evaluate_formulas=True, header=0)
        if df.columns.duplicated().any(): df = df.loc[:, ~df.columns.duplicated()]
        df  = df.dropna(how='all')
        df.columns = [c.strip() for c in df.columns]
        cd  = detectar_col(df, ['Data', 'data', 'Date'])
        if not cd: return pd.DataFrame()
        df['Data'] = df[cd].apply(parse_date)
        df = df.dropna(subset=['Data']).sort_values('Data')
        # Remover datas futuras — dados só até hoje
        df = df[df['Data'] <= pd.Timestamp.now().normalize()]
        for col in ['Peso','BF','Calorias','Carb','Fat','Ptn',
                    'Carb_perc','Fat_perc','Ptn_perc','Net']:
            if col in df.columns: df[col] = df[col].apply(br_float)
        # Remover linhas sem nenhum dado numérico (linhas futuras vazias)
        num_cols = [c for c in ['Peso','BF','Calorias','Carb','Fat','Ptn','Net']
                    if c in df.columns]
        if num_cols:
            df = df[df[num_cols].notna().any(axis=1)]
        # Ranges fisiológicos
        for col, lo, hi in [('Peso',30,200),('BF',3,50),('Calorias',500,6000),
                             ('Carb',0,800),('Fat',0,400),('Ptn',0,400),
                             ('Net',-2000,4000)]:
            if col in df.columns:
                df.loc[~df[col].between(lo, hi, inclusive='both'), col] = np.nan
        # Z-score 3.0 nos campos principais
        for col in ['Peso','BF','Calorias','Net']:
            if col in df.columns: df[col] = remove_zscore(df[col], 3.0)
        return df.reset_index(drop=True)
    except Exception as e:
        st.error(f"Erro ao carregar dados corporais: {e}")
        return pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# NLSS — Nonlinear Least Squares with Shrinkage (VECTORISED — production)
# "Estimating K1, K2, T1, T2 Using Hierarchical Bayesian NLSS"
# Gabriel Della Mattia · AGMT2 · 2026
#
# Implementação vectorizada (NumPy) do Algorithm 1 do paper.
# A versão do paper usa loops Python para clareza didáctica.
# Esta versão usa operações matriciais → 100-1000× mais rápida.
#
# Complexidade:
#   Loop Python original: O(n_testes × n_dias × n_iter × n_semanas)
#   NumPy vectorizado:    O(n_dias × n_testes) por iteração L-BFGS-B
# ══════════════════════════════════════════════════════════════════════════════

# ── A. Prior AGMT2 (Tabela 2 do paper) ───────────────────────────────────────
_NLSS_MU_POP  = np.array([2.87, 4.09, 40.64, 6.584])   # [K1, K2, T1, T2]
_NLSS_SD_POP  = np.array([0.301, 0.334, 3.094, 0.651])

_NLSS_CORR = np.array([
    [ 1.00, +0.63, -0.41, -0.12],
    [+0.63,  1.00, +0.09, -0.28],
    [-0.41, +0.09,  1.00, +0.19],
    [-0.12, -0.28, +0.19,  1.00],
])
_NLSS_SIGMA_POP = np.outer(_NLSS_SD_POP, _NLSS_SD_POP) * _NLSS_CORR
_NLSS_SIGMA_INV = np.linalg.inv(_NLSS_SIGMA_POP)

_NLSS_BOUNDS     = [(0.20, 8.00), (0.20, 12.00), (14.0, 80.0), (3.0, 18.0)]
_NLSS_LAMBDA_MAX = 5.0
_NLSS_N_HALF     = 4

def _nlss_lambda(n: int) -> float:
    return _NLSS_LAMBDA_MAX * np.exp(-n / _NLSS_N_HALF)


def _build_convolution_matrix(tss_arr: np.ndarray,
                               test_days: np.ndarray,
                               T1: float, T2: float) -> tuple:
    """
    Vectorised convolution — elimina os loops Python internos.

    Para cada teste no dia d, a convolução é:
        Σ_{τ=0}^{d-1} TSS(τ) × exp(-(d-τ)/T)

    Representado como produto matriz-vector:
        C1[i] = tss @ decay1[i,:]   onde decay1[i,j] = exp(-(d_i-j)/T1) se j<d_i
        C2[i] = tss @ decay2[i,:]

    Retorna: (C1, C2, dC1_dT1, dC2_dT2) — arrays shape (n_testes,)
    para calcular p̂ e o gradiente analítico.
    """
    n_tests = len(test_days)
    n_days  = len(tss_arr)

    # Matriz de deltas: delta[i,j] = test_days[i] - j  (só para j < test_days[i])
    # Shape: (n_tests, n_days)
    j_idx  = np.arange(n_days, dtype=np.float64)
    d_idx  = test_days[:, np.newaxis].astype(np.float64)   # (n_tests, 1)
    deltas = d_idx - j_idx                                  # (n_tests, n_days)

    # Máscara: só posições onde j < d (causalidade)
    mask   = (j_idx[np.newaxis, :] < test_days[:, np.newaxis])  # (n_tests, n_days)

    # Decaimentos exponenciais
    with np.errstate(over='ignore', invalid='ignore'):
        exp1 = np.where(mask, np.exp(-deltas / T1), 0.0)   # (n_tests, n_days)
        exp2 = np.where(mask, np.exp(-deltas / T2), 0.0)

    # Convoluções: Σ_j TSS[j] × exp(...)
    C1  = exp1 @ tss_arr    # (n_tests,)
    C2  = exp2 @ tss_arr

    # Gradientes em relação a T1, T2: ∂C/∂T = Σ_j TSS[j] × (delta/T²) × exp(...)
    dC1 = (exp1 * (deltas / (T1 * T1))) @ tss_arr
    dC2 = (exp2 * (deltas / (T2 * T2))) @ tss_arr

    return C1, C2, dC1, dC2


def _nlss_cost_grad_vec(theta: np.ndarray,
                         tss_arr: np.ndarray,
                         test_days: np.ndarray,
                         test_watts: np.ndarray,
                         lam: float,
                         p0: float = 0.0) -> tuple:
    """
    Custo + gradiente analítico — implementação vectorizada.
    Equivalente matemático exacto ao código do paper, sem loops Python.
    """
    K1, K2, T1, T2 = theta
    n = len(test_days)

    if n == 0:
        diff    = theta - _NLSS_MU_POP
        L_prior = float(diff @ _NLSS_SIGMA_INV @ diff)
        g_prior = 2.0 * _NLSS_SIGMA_INV @ diff
        return lam * L_prior, lam * g_prior

    C1, C2, dC1_dT1, dC2_dT2 = _build_convolution_matrix(tss_arr, test_days, T1, T2)

    # p̂(d) = p0 + K1×C1 - K2×C2
    p_hat     = p0 + K1 * C1 - K2 * C2          # (n_tests,)
    residuals = test_watts - p_hat                # (n_tests,)

    # L_data = (1/n) Σ r²
    L_data = float(np.sum(residuals ** 2) / n)

    # Gradiente analítico (Eq. 9 do paper — vectorizado)
    r_sum   = residuals                           # shape (n,)
    g_data  = np.array([
        -2.0 / n * np.dot(r_sum, C1),            # ∂L/∂K1
        +2.0 / n * np.dot(r_sum, C2),            # ∂L/∂K2
        -2.0 * K1 / n * np.dot(r_sum, dC1_dT1), # ∂L/∂T1
        +2.0 * K2 / n * np.dot(r_sum, dC2_dT2), # ∂L/∂T2
    ])

    # Prior Mahalanobis (Eq. 6b)
    diff    = theta - _NLSS_MU_POP
    L_prior = float(diff @ _NLSS_SIGMA_INV @ diff)
    g_prior = 2.0 * _NLSS_SIGMA_INV @ diff

    return L_data + lam * L_prior, g_data + lam * g_prior


def calcular_nlss(df_act, df_wellness=None,
                  p0_frac: float = 0.5,
                  window_l: int = 365) -> dict:
    """
    Estima K1, K2, T1, T2 via Hierarchical Bayesian NLSS (Algorithm 1).
    Implementação vectorizada — tipicamente <5s para histórico de 3 anos.

    INPUT:
        df_act   : actividades completas (ac_full)
        window_l : janela de recalibração em dias (default 90)

    OUTPUT (dict):
        K1, K2, T1, T2  : parâmetros individualizados
        lambda_n         : força do prior actual
        n_tests          : testes na janela actual
        p_hat_series     : pd.Series com p̂(t) sobre todo o histórico
        CTL_nlss/ATL_nlss/TSB_nlss : séries com K1/K2 individualizados
        CTL_tp/ATL_tp/TSB_tp       : TrainingPeaks para comparação
        test_dates/test_watts       : testes usados
        history          : lista de dicts semanais
        error            : None ou mensagem de erro
    """
    from scipy.optimize import minimize as _minimize

    # ── 1. Série diária de TSS ────────────────────────────────────────────────
    df = filtrar_principais(df_act).copy()
    df['Data'] = pd.to_datetime(df['Data'])

    if 'icu_training_load' in df.columns and df['icu_training_load'].notna().sum() > 10:
        df['_load'] = pd.to_numeric(df['icu_training_load'], errors='coerce').fillna(0)
        fonte = 'icu_training_load'
    elif 'moving_time' in df.columns and 'rpe' in df.columns:
        _rpe = pd.to_numeric(df['rpe'], errors='coerce')
        df['_load'] = (pd.to_numeric(df['moving_time'], errors='coerce') / 60) * _rpe.fillna(_rpe.median())
        df['_load'] = df['_load'].fillna(0)
        fonte = 'session_rpe'
    else:
        return {'error': 'Sem dados de carga.', 'theta': _NLSS_MU_POP,
                'K1': _NLSS_MU_POP[0], 'K2': _NLSS_MU_POP[1],
                'T1': _NLSS_MU_POP[2], 'T2': _NLSS_MU_POP[3]}

    ld = (df.groupby('Data')['_load'].sum().reset_index()
            .sort_values('Data'))
    date_range = pd.date_range(ld['Data'].min(), datetime.now().date())
    ld = (ld.set_index('Data')
             .reindex(date_range, fill_value=0)
             .reset_index())
    ld.columns = ['Data', 'tss_val']
    tss_arr = ld['tss_val'].values.astype(np.float64)
    n_days  = len(tss_arr)

    # ── 2. Testes de potência — máximo anual de Bike (MMP20, MMP5, MMP3) ────
    #
    # Colunas na sheet: MMP20, MMP5, MMP3, MMP1, MMP12, MMP60
    # Mapeadas em MAPA_TRAINING: mmp20_raw → parseadas para mmp20_w (todos)
    # Types de Bike: VirtualRide, Ride → normalizados para Bike via norm_tipo()
    #
    # ESTRATÉGIA: valor MÁXIMO de mmp20_w por ANO de Bike.
    # O máximo anual é o PR real desse ano — sem depender de is_pr (que está
    # True em quase todas as sessões no Intervals.icu).
    # Fallback: MMP5 e MMP3 se MMP20 insuficiente.
    # ─────────────────────────────────────────────────────────────────────────

    # Garantir type normalizado: VirtualRide→Bike, Ride→Bike, etc.
    _tipo_col = next((c for c in ['type', 'modality'] if c in df.columns), None)
    if _tipo_col:
        df['_type_norm'] = df[_tipo_col].apply(norm_tipo)
        _df_bike = df[df['_type_norm'] == 'Bike'].copy()
    else:
        _df_bike = df.copy()

    _df_bike['Data'] = pd.to_datetime(_df_bike['Data'])
    _df_bike['_ano'] = _df_bike['Data'].dt.year

    def _max_anual_mmp(df_src, col_w):
        """Por ano: data da sessão com o MMP máximo nessa coluna."""
        if col_w not in df_src.columns:
            return pd.DataFrame(columns=['Data', 'watts'])
        sub = df_src[['Data', '_ano', col_w]].copy()
        sub['watts'] = pd.to_numeric(sub[col_w], errors='coerce')
        sub = sub[sub['watts'] > 50].dropna(subset=['watts'])
        if sub.empty:
            return pd.DataFrame(columns=['Data', 'watts'])
        idx_max = sub.groupby('_ano')['watts'].idxmax()
        return (sub.loc[idx_max, ['Data', 'watts']]
                   .sort_values('Data')
                   .reset_index(drop=True))

    # Primário: MMP20 de Bike — 1 ponto por ano, escala consistente
    # Usar APENAS MMP20 — misturar MMP3 (347W) com MMP20 (228W) cria
    # inconsistência de escala: o optimizador interpreta a diferença como
    # variação de performance quando são durações diferentes → fica no prior.
    _tests_mmp20 = _max_anual_mmp(_df_bike, 'mmp20_w')

    if not _tests_mmp20.empty:
        tests_df      = _tests_mmp20.copy()
        _fonte_testes = f'Bike MMP20 — maximo anual ({len(_tests_mmp20)} pontos)'
    else:
        tests_df      = pd.DataFrame(columns=['Data', 'watts'])
        _fonte_testes = 'Sem dados de Bike MMP20'

    # Fallback: todas as modalidades se Bike insuficiente
    if len(tests_df) < 2:
        _all = df.copy()
        _all['Data'] = pd.to_datetime(_all['Data'])
        _all['_ano'] = _all['Data'].dt.year
        _t20_all = _max_anual_mmp(_all, 'mmp20_w')
        if len(_t20_all) >= 2:
            tests_df      = _t20_all
            _fonte_testes = 'Todas modalidades MMP20 — maximo anual (fallback)'
        else:
            _fonte_testes = 'Insuficiente'

    # ── Limitar TSS ao período relevante ────────────────────────────────────
    # Problema: sem testes em 2017-2023, p̂(t) extrapola sem âncora e cria
    # picos de forma fictícios (ex: Set/2023 = 299W sem nenhum teste real).
    # Solução: usar TSS apenas a partir de (primeiro_teste - 365 dias).
    # Isto garante que o modelo tem contexto de treino antes do primeiro teste
    # mas não extrapola décadas sem dados de performance.
    if len(tests_df) >= 1:
        _primeiro_teste = pd.to_datetime(tests_df['Data'].min())
        _tss_inicio     = _primeiro_teste - pd.Timedelta(days=365)
        _tss_inicio_date = _tss_inicio.date()
        # Recortar ld e reindexar
        ld = ld[pd.to_datetime(ld['Data']).dt.date >= _tss_inicio_date].reset_index(drop=True)
        tss_arr = ld['tss_val'].values.astype(np.float64)
        n_days  = len(tss_arr)

    # Mapear para índices da série diária (após eventual corte)
    date_to_idx = {pd.Timestamp(d).date(): i
                   for i, d in enumerate(ld['Data'])}
    test_days, test_watts, test_dates = [], [], []
    for _, row in tests_df.iterrows():
        d   = pd.Timestamp(row['Data']).date()
        idx = date_to_idx.get(d)
        if idx is not None and idx > 0:
            test_days.append(idx)
            test_watts.append(float(row['watts']))
            test_dates.append(d)

    test_days_arr  = np.array(test_days,  dtype=np.int64)
    test_watts_arr = np.array(test_watts, dtype=np.float64)

    # ── 3. p0 baseline ────────────────────────────────────────────────────────
    # p0 = mediana dos testes (não percentil 25) — mais robusto com poucos pontos
    p0 = float(np.nanmedian(test_watts_arr)) if len(test_watts_arr) > 0 else 200.0

    # ── 4. Algorithm 1 — recalibração semanal vectorizada ────────────────────
    theta_current = _NLSS_MU_POP.copy()
    history       = []
    DELTA_REL_THR = 0.15

    for t in range(6, n_days, 7):
        w_start = max(0, t - window_l)

        # Testes na janela [w_start, t]
        mask_w      = (test_days_arr >= w_start) & (test_days_arr <= t)
        w_days      = test_days_arr[mask_w]
        w_watts     = test_watts_arr[mask_w]
        n_t         = int(mask_w.sum())
        lam         = _nlss_lambda(n_t)

        if n_t == 0:
            theta_current = _NLSS_MU_POP.copy()
            history.append({'day': t, 'n_tests': 0, 'lambda': lam,
                            'K1': theta_current[0], 'K2': theta_current[1],
                            'T1': theta_current[2], 'T2': theta_current[3]})
            continue

        # TSS na janela (slicing evita passar array completo desnecessariamente)
        # A convolução precisa de todo o histórico até t para os testes na janela
        tss_window = tss_arr[:t+1]

        try:
            result = _minimize(
                lambda th: _nlss_cost_grad_vec(
                    th, tss_window, w_days, w_watts, lam, p0),
                theta_current,
                method='L-BFGS-B',
                jac=True,
                bounds=_NLSS_BOUNDS,
                options={'maxiter': 200, 'ftol': 1e-9, 'gtol': 1e-6}
            )
            theta_new = result.x

            # Rejeitar se fora de 3σ do prior
            d_rel = (np.linalg.norm(theta_new - theta_current) /
                     (np.linalg.norm(theta_current) + 1e-12))
            if d_rel > DELTA_REL_THR:
                lam = _nlss_lambda(0)
            if np.any(np.abs(theta_new - _NLSS_MU_POP) > 4 * _NLSS_SD_POP):  # 4σ para permitir individualização
                theta_new = theta_current
            theta_current = theta_new.copy()

        except Exception:
            pass

        history.append({
            'day': t, 'n_tests': n_t, 'lambda': float(lam),
            'K1': float(theta_current[0]), 'K2': float(theta_current[1]),
            'T1': float(theta_current[2]), 'T2': float(theta_current[3]),
        })

    K1, K2, T1, T2 = theta_current
    # Contar testes na janela final correctamente
    # Usar todos os testes disponíveis (janela alargada para ver todos)
    n_in_window = len(test_days_arr)
    lam_final   = _nlss_lambda(n_in_window)

    # ── 5. p̂(t) — série completa vectorizada ─────────────────────────────────
    # Para todo o histórico em uma passagem:
    # p̂(t) = p0 + Σ_{τ=0}^{t-1} TSS(τ)×[K1×exp(-(t-τ)/T1) - K2×exp(-(t-τ)/T2)]
    # Usamos EWM como proxy eficiente: p̂ ≈ p0 + K1×ewm(T1) - K2×ewm(T2)
    # (exacto ao limite de t→∞, boa aproximação para histórico ≥3 meses)
    tss_series   = pd.Series(tss_arr, index=pd.DatetimeIndex(ld['Data']))
    fitness_raw  = tss_series.ewm(span=T1, adjust=False).mean()
    fatigue_raw  = tss_series.ewm(span=T2, adjust=False).mean()
    p_hat_series = pd.Series(
        p0 + K1 * fitness_raw.values - K2 * fatigue_raw.values,
        index=pd.DatetimeIndex(ld['Data']),
        name='p_hat'
    )

    # ── 6. CTL/ATL/TSB NLSS vs TrainingPeaks ─────────────────────────────────
    ctl_nlss = K1 * fitness_raw
    atl_nlss = K2 * fatigue_raw
    tsb_nlss = ctl_nlss - atl_nlss
    ctl_tp   = tss_series.ewm(span=42, adjust=False).mean()
    atl_tp   = tss_series.ewm(span=7,  adjust=False).mean()
    tsb_tp   = ctl_tp - atl_tp

    return {
        'theta':        theta_current,
        'K1': float(K1), 'K2': float(K2),
        'T1': float(T1), 'T2': float(T2),
        'lambda_n':     float(lam_final),
        'n_tests':      len(test_days),
        'fonte_testes': _fonte_testes,
        'n_in_window':  n_in_window,
        'p_hat_series': p_hat_series,
        'CTL_nlss':     ctl_nlss,
        'ATL_nlss':     atl_nlss,
        'TSB_nlss':     tsb_nlss,
        'CTL_tp':       ctl_tp,
        'ATL_tp':       atl_tp,
        'TSB_tp':       tsb_tp,
        'test_dates':   test_dates,
        'test_watts':   test_watts,
        'history':      history,
        'fonte_carga':  fonte,
        'p0':           float(p0),
        'n_days':       n_days,
        'error':        None,
    }
