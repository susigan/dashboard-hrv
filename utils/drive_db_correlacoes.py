# ══════════════════════════════════════════════════════════════════════════════
# utils/drive_db_correlacoes.py — Guardar análises de correlacoes no Drive DB
# Versão 2: Usa IDs diretos dos ficheiros (não pasta)
# ══════════════════════════════════════════════════════════════════════════════

import os, io, sqlite3, streamlit as st, pandas as pd
from datetime import datetime
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from google.oauth2.service_account import Credentials

# ── CONFIGURAÇÃO ──────────────────────────────────────────────────────────────
_DB_NAME         = "correlacoes.db"
_LOCAL_DB        = f"/tmp/{_DB_NAME}"
_DB_FILE_ID      = "1K2c3Xl77XIqf7AYpnXn4fzqWfLE99CVf"  # ← ID direto do ficheiro
_DRIVE_SCOPES = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive",
]

def _drive_svc():
    """Conexão ao Google Drive"""
    try:
        creds = Credentials.from_service_account_info(
            dict(st.secrets["gcp_service_account"]), scopes=_DRIVE_SCOPES)
        return build("drive", "v3", credentials=creds)
    except Exception as e:
        st.error(f"Erro ao conectar ao Drive: {e}")
        return None

def _init_correlacoes_db():
    """Inicializa DB com tabelas"""
    try:
        conn = sqlite3.connect(_LOCAL_DB)
        c = conn.cursor()
        c.executescript("""
        CREATE TABLE IF NOT EXISTS correlacoes_rpe (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            saved_at TEXT, data_treino TEXT, modalidade TEXT,
            rpe_categoria TEXT, rpe_valor REAL,
            hrv_baseline REAL, hrv_lag1 REAL, hrv_lag2 REAL, hrv_lag3 REAL,
            hrv_lag4 REAL, hrv_lag5 REAL, hrv_lag7 REAL,
            delta_lag1_pct REAL, delta_lag2_pct REAL, delta_lag3_pct REAL,
            delta_lag4_pct REAL, delta_lag5_pct REAL, delta_lag7_pct REAL,
            n_samples INTEGER, notas TEXT);
        CREATE TABLE IF NOT EXISTS correlacoes_kj (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            saved_at TEXT, data_treino TEXT, modalidade TEXT,
            kj_valor REAL, kj_quartil TEXT,
            hrv_baseline REAL, hrv_lag1 REAL, hrv_lag2 REAL, hrv_lag3 REAL,
            hrv_lag4 REAL, hrv_lag5 REAL, hrv_lag7 REAL,
            delta_lag1_pct REAL, delta_lag2_pct REAL, delta_lag3_pct REAL,
            delta_lag4_pct REAL, delta_lag5_pct REAL, delta_lag7_pct REAL,
            n_samples INTEGER, notas TEXT);
        CREATE TABLE IF NOT EXISTS correlacoes_tipo_kj (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            saved_at TEXT, data_treino TEXT, modalidade TEXT,
            tipo_atividade TEXT, kj_quartil TEXT,
            hrv_baseline REAL, hrv_lag1 REAL, hrv_lag2 REAL, hrv_lag3 REAL,
            hrv_lag4 REAL, hrv_lag5 REAL, hrv_lag7 REAL,
            delta_lag1_pct REAL, delta_lag2_pct REAL, delta_lag3_pct REAL,
            delta_lag4_pct REAL, delta_lag5_pct REAL, delta_lag7_pct REAL,
            n_samples INTEGER, notas TEXT);
        """)
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Erro ao inicializar DB: {e}")
        return False

def _upload_correlacoes_db() -> bool:
    """Upload do DB para Drive usando ID direto"""
    if not os.path.exists(_LOCAL_DB):
        st.warning("Ficheiro local não existe")
        return False
    try:
        svc = _drive_svc()
        if not svc:
            return False
        
        media = MediaFileUpload(_LOCAL_DB, mimetype="application/x-sqlite3", resumable=False)
        svc.files().update(
            fileId=_DB_FILE_ID,
            media_body=media,
            supportsAllDrives=True,
        ).execute()
        st.success("✅ DB atualizado no Drive!")
        return True
    except Exception as e:
        st.error(f"Erro ao fazer upload: {e}")
        return False

def _download_correlacoes_db() -> bool:
    """Download do DB do Drive usando ID direto"""
    try:
        svc = _drive_svc()
        if not svc:
            return False
        
        req = svc.files().get_media(fileId=_DB_FILE_ID, supportsAllDrives=True)
        with open(_LOCAL_DB, "wb") as f:
            dl = MediaIoBaseDownload(f, req)
            done = False
            while not done:
                _, done = dl.next_chunk()
        
        st.success("✅ DB carregado do Drive!")
        return True
    except Exception as e:
        st.warning(f"Download falhou, usando DB local: {e}")
        return False

def get_correlacoes_conn() -> sqlite3.Connection:
    """Retorna conexão SQLite"""
    if not os.path.exists(_LOCAL_DB):
        _download_correlacoes_db()
    _init_correlacoes_db()
    return sqlite3.connect(_LOCAL_DB)

# ── FUNÇÕES DE GRAVAÇÃO ───────────────────────────────────────────────────────

def save_correlacoes_rpe(data_treino: str, modalidade: str, rpe_categoria: str,
                         rpe_valor: float, hrv_baseline: float,
                         hrv_lags: dict, delta_pcts: dict, n_samples: int = 0,
                         notas: str = "") -> bool:
    """Guarda análise RPE no DB e faz upload"""
    try:
        conn = get_correlacoes_conn()
        c = conn.cursor()
        c.execute("""
            INSERT INTO correlacoes_rpe
            (saved_at, data_treino, modalidade, rpe_categoria, rpe_valor,
             hrv_baseline, hrv_lag1, hrv_lag2, hrv_lag3, hrv_lag4, hrv_lag5, hrv_lag7,
             delta_lag1_pct, delta_lag2_pct, delta_lag3_pct, delta_lag4_pct,
             delta_lag5_pct, delta_lag7_pct, n_samples, notas)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().strftime("%Y-%m-%d %H:%M"),
            data_treino, modalidade, rpe_categoria, float(rpe_valor) if rpe_valor else 0,
            float(hrv_baseline) if hrv_baseline else 0,
            float(hrv_lags.get('lag1', 0) or 0), float(hrv_lags.get('lag2', 0) or 0),
            float(hrv_lags.get('lag3', 0) or 0), float(hrv_lags.get('lag4', 0) or 0),
            float(hrv_lags.get('lag5', 0) or 0), float(hrv_lags.get('lag7', 0) or 0),
            float(delta_pcts.get('delta1', 0) or 0), float(delta_pcts.get('delta2', 0) or 0),
            float(delta_pcts.get('delta3', 0) or 0), float(delta_pcts.get('delta4', 0) or 0),
            float(delta_pcts.get('delta5', 0) or 0), float(delta_pcts.get('delta7', 0) or 0),
            n_samples, notas
        ))
        conn.commit()
        conn.close()
        
        return _upload_correlacoes_db()
    except Exception as e:
        st.error(f"Erro ao guardar correlacoes_rpe: {e}")
        return False

def save_correlacoes_kj(data_treino: str, modalidade: str, kj_valor: float,
                       kj_quartil: str, hrv_baseline: float,
                       hrv_lags: dict, delta_pcts: dict, n_samples: int = 0,
                       notas: str = "") -> bool:
    """Guarda análise kJ no DB e faz upload"""
    try:
        conn = get_correlacoes_conn()
        c = conn.cursor()
        c.execute("""
            INSERT INTO correlacoes_kj
            (saved_at, data_treino, modalidade, kj_valor, kj_quartil,
             hrv_baseline, hrv_lag1, hrv_lag2, hrv_lag3, hrv_lag4, hrv_lag5, hrv_lag7,
             delta_lag1_pct, delta_lag2_pct, delta_lag3_pct, delta_lag4_pct,
             delta_lag5_pct, delta_lag7_pct, n_samples, notas)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().strftime("%Y-%m-%d %H:%M"),
            data_treino, modalidade, float(kj_valor) if kj_valor else 0, kj_quartil,
            float(hrv_baseline) if hrv_baseline else 0,
            float(hrv_lags.get('lag1', 0) or 0), float(hrv_lags.get('lag2', 0) or 0),
            float(hrv_lags.get('lag3', 0) or 0), float(hrv_lags.get('lag4', 0) or 0),
            float(hrv_lags.get('lag5', 0) or 0), float(hrv_lags.get('lag7', 0) or 0),
            float(delta_pcts.get('delta1', 0) or 0), float(delta_pcts.get('delta2', 0) or 0),
            float(delta_pcts.get('delta3', 0) or 0), float(delta_pcts.get('delta4', 0) or 0),
            float(delta_pcts.get('delta5', 0) or 0), float(delta_pcts.get('delta7', 0) or 0),
            n_samples, notas
        ))
        conn.commit()
        conn.close()
        
        return _upload_correlacoes_db()
    except Exception as e:
        st.error(f"Erro ao guardar correlacoes_kj: {e}")
        return False

def load_correlacoes_history(table: str = "correlacoes_rpe", n: int = 50) -> pd.DataFrame:
    """Carrega histórico do DB"""
    try:
        conn = get_correlacoes_conn()
        df = pd.read_sql_query(f"SELECT * FROM {table} ORDER BY saved_at DESC LIMIT {n}", conn)
        conn.close()
        return df
    except:
        return pd.DataFrame()
