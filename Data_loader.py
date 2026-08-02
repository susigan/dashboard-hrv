# data_loader.py — ATHELTICA Dashboard

import streamlit as st
import pandas as pd
import numpy as np
import gspread
from gspread_dataframe import get_as_dataframe
from google.oauth2.service_account import Credentials
from datetime import datetime, timedelta

from utils.config import (WELLNESS_URL, TRAINING_URL, SCOPES,
                    MAPA_WELLNESS, MAPA_TRAINING, VALID_TYPES, TYPE_MAP,
                    ANNUAL_SPREADSHEET_ID, ANNUAL_SHEETS)
from utils.data import (detectar_col, br_float, parse_date, norm_tipo,
                            remove_zscore, remove_zeros, fill_missing)

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
                           'stress', 'fatiga', 'humor', 'soreness', 'peso', 'fat',
                           'hf_power']
              if c in df.columns]:
        df[c] = remove_zscore(df[c], 3.0)
    # [2] Zeros inválidos → NaN
    df = remove_zeros(df, ['hrv', 'rhr', 'sleep_hours', 'hf_power'])
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
        for col in ['Peso','BF','Calorias','Carb','Fat','Ptn',
                    'Carb_perc','Fat_perc','Ptn_perc','Net']:
            if col in df.columns: df[col] = df[col].apply(br_float)
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
