# ══════════════════════════════════════════════════════════════════════════════
# utils/drive_db_hrv_analyzer.py — Guardar análises HRV Analyzer no Drive DB
# ══════════════════════════════════════════════════════════════════════════════

import os, io, sqlite3, streamlit as st, pandas as pd
from datetime import datetime
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from google.oauth2.service_account import Credentials

# ── CONFIGURAÇÃO ──────────────────────────────────────────────────────────────
_DB_NAME     = "hrv_analyzer.db"
_LOCAL_DB    = f"/tmp/{_DB_NAME}"
_FOLDER_ID   = "11oXQPkFrG6ZBCsvjDqb8RAiE_VfwBSfV"  # ← SUBSTITUIR PELO TEU FOLDER_ID
_DRIVE_SCOPES = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive",
]

def _drive_svc():
    """Conexão ao Google Drive"""
    creds = Credentials.from_service_account_info(
        dict(st.secrets["gcp_service_account"]), scopes=_DRIVE_SCOPES)
    return build("drive", "v3", credentials=creds)

def _find_db_id(svc) -> str | None:
    """Procura o DB no Drive"""
    try:
        r = svc.files().list(
            q=f"name='{_DB_NAME}' and '{_FOLDER_ID}' in parents and trashed=false",
            fields="files(id)",
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ).execute()
        files = r.get("files", [])
        return files[0]["id"] if files else None
    except:
        return None

def _init_hrv_db():
    """Inicializa DB com tabelas"""
    conn = sqlite3.connect(_LOCAL_DB)
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS hrv_analysis_daily (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        saved_at TEXT, data_wellness TEXT,
        hrv_valor REAL, rhr_valor REAL, sono_horas REAL, stress_nivel INTEGER,
        wellness_score REAL, recuperacao_pattern TEXT,
        hrv_guided_suggestion TEXT, javaloyes_status TEXT,
        javaloyes_swc_inferior REAL, javaloyes_swc_superior REAL,
        kiviniemi_status TEXT, kiviniemi_swc_inferior REAL, kiviniemi_swc_superior REAL,
        baseline_lag60 REAL, desvio_baseline_pct REAL, notas TEXT);
    CREATE TABLE IF NOT EXISTS hrv_recovery_patterns (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        saved_at TEXT, periodo_inicio TEXT, periodo_fim TEXT,
        dias_analisados INTEGER, hrv_media REAL, hrv_std REAL,
        hrv_min REAL, hrv_max REAL, rhr_media REAL, sono_media REAL,
        stress_media REAL, padroes_identificados TEXT,
        recomendacoes TEXT, notas TEXT);
    CREATE TABLE IF NOT EXISTS hrv_lag_analysis (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        saved_at TEXT, variavel_independente TEXT, lag_dias INTEGER,
        correlacao_r REAL, p_value REAL, n_amostras INTEGER,
        periodo_analise TEXT, notas TEXT);
    CREATE TABLE IF NOT EXISTS hrv_baseline_tracking (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        saved_at TEXT, data_calculo TEXT,
        baseline_60d REAL, baseline_90d REAL, baseline_180d REAL,
        variacao_60d_pct REAL, variacao_90d_pct REAL, variacao_180d_pct REAL,
        tendencia TEXT, notas TEXT);
    """)
    conn.commit()
    conn.close()

def _upload_hrv_db() -> bool:
    """Upload do DB para Drive"""
    if not os.path.exists(_LOCAL_DB):
        return False
    try:
        svc = _drive_svc()
        file_id = _find_db_id(svc)
        media = MediaFileUpload(_LOCAL_DB, mimetype="application/x-sqlite3", resumable=False)
        if file_id:
            svc.files().update(fileId=file_id, media_body=media, supportsAllDrives=True).execute()
        else:
            svc.files().create(
                body={"name": _DB_NAME, "parents": [_FOLDER_ID]},
                media_body=media, supportsAllDrives=True, fields="id").execute()
        return True
    except:
        return False

def _download_hrv_db() -> bool:
    """Download do DB do Drive"""
    try:
        svc = _drive_svc()
        file_id = _find_db_id(svc)
        if file_id:
            req = svc.files().get_media(fileId=file_id, supportsAllDrives=True)
            with open(_LOCAL_DB, "wb") as f:
                dl = MediaIoBaseDownload(f, req)
                done = False
                while not done:
                    _, done = dl.next_chunk()
    except:
        pass
    _init_hrv_db()
    return True

def get_hrv_conn() -> sqlite3.Connection:
    """Retorna conexão SQLite"""
    if not os.path.exists(_LOCAL_DB):
        _download_hrv_db()
    _init_hrv_db()
    return sqlite3.connect(_LOCAL_DB)

# ── FUNÇÕES DE GRAVAÇÃO ───────────────────────────────────────────────────────

def save_hrv_daily_analysis(data_wellness: str, hrv: float, rhr: float,
                            sono_horas: float, stress: int, wellness_score: float,
                            recuperacao_pattern: str = "",
                            hrv_guided_suggestion: str = "",
                            javaloyes_status: str = "",
                            javaloyes_swc: dict = None,
                            kiviniemi_status: str = "",
                            kiviniemi_swc: dict = None,
                            baseline_lag60: float = 0,
                            notas: str = "") -> bool:
    """Guarda análise HRV diária"""
    try:
        conn = get_hrv_conn()
        c = conn.cursor()
        c.execute("""
            INSERT INTO hrv_analysis_daily
            (saved_at, data_wellness, hrv_valor, rhr_valor, sono_horas, stress_nivel,
             wellness_score, recuperacao_pattern, hrv_guided_suggestion,
             javaloyes_status, javaloyes_swc_inferior, javaloyes_swc_superior,
             kiviniemi_status, kiviniemi_swc_inferior, kiviniemi_swc_superior,
             baseline_lag60, desvio_baseline_pct, notas)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().strftime("%Y-%m-%d %H:%M"),
            data_wellness, hrv, rhr, sono_horas, stress,
            wellness_score, recuperacao_pattern, hrv_guided_suggestion,
            javaloyes_status,
            javaloyes_swc.get('inferior', 0) if javaloyes_swc else 0,
            javaloyes_swc.get('superior', 0) if javaloyes_swc else 0,
            kiviniemi_status,
            kiviniemi_swc.get('inferior', 0) if kiviniemi_swc else 0,
            kiviniemi_swc.get('superior', 0) if kiviniemi_swc else 0,
            baseline_lag60,
            ((hrv - baseline_lag60) / baseline_lag60 * 100) if baseline_lag60 > 0 else 0,
            notas
        ))
        conn.commit()
        conn.close()
        
        _upload_hrv_db()
        return True
    except Exception as e:
        st.error(f"Erro ao guardar hrv_daily: {e}")
        return False

def save_recovery_patterns(periodo_inicio: str, periodo_fim: str,
                          dias_analisados: int, hrv_stats: dict,
                          rhr_media: float, sono_media: float, stress_media: float,
                          padroes_identificados: str = "",
                          recomendacoes: str = "",
                          notas: str = "") -> bool:
    """Guarda análise de padrões de recuperação"""
    try:
        conn = get_hrv_conn()
        c = conn.cursor()
        c.execute("""
            INSERT INTO hrv_recovery_patterns
            (saved_at, periodo_inicio, periodo_fim, dias_analisados,
             hrv_media, hrv_std, hrv_min, hrv_max, rhr_media, sono_media,
             stress_media, padroes_identificados, recomendacoes, notas)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().strftime("%Y-%m-%d %H:%M"),
            periodo_inicio, periodo_fim, dias_analisados,
            hrv_stats.get('media', 0), hrv_stats.get('std', 0),
            hrv_stats.get('min', 0), hrv_stats.get('max', 0),
            rhr_media, sono_media, stress_media,
            padroes_identificados, recomendacoes, notas
        ))
        conn.commit()
        conn.close()
        
        _upload_hrv_db()
        return True
    except Exception as e:
        st.error(f"Erro ao guardar recovery_patterns: {e}")
        return False

def load_hrv_history(table: str = "hrv_analysis_daily", n: int = 50) -> pd.DataFrame:
    """Carrega histórico do DB"""
    try:
        conn = get_hrv_conn()
        df = pd.read_sql_query(f"SELECT * FROM {table} ORDER BY saved_at DESC LIMIT {n}", conn)
        conn.close()
        return df
    except:
        return pd.DataFrame()
