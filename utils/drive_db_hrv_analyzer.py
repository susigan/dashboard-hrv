# ══════════════════════════════════════════════════════════════════════════════
# utils/drive_db_hrv_analyzer.py — Guardar análises HRV Analyzer no Drive DB
# Versão 2: Usa IDs diretos dos ficheiros (não pasta)
# ══════════════════════════════════════════════════════════════════════════════

import os, io, sqlite3, streamlit as st, pandas as pd
from datetime import datetime
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from google.oauth2.service_account import Credentials

# ── CONFIGURAÇÃO ──────────────────────────────────────────────────────────────
_DB_NAME         = "hrv_analyzer.db"
_LOCAL_DB        = f"/tmp/{_DB_NAME}"
_DB_FILE_ID      = "1SoP0W0qGdpkzhB177tSSBmnpj6-BdlM3"  # ← ID direto do ficheiro
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

def _init_hrv_db():
    """Inicializa DB com tabelas"""
    try:
        conn = sqlite3.connect(_LOCAL_DB)
        c = conn.cursor()
        c.executescript("""
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
        return True
    except Exception as e:
        st.error(f"Erro ao inicializar DB: {e}")
        return False

def _upload_hrv_db() -> bool:
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

def _download_hrv_db() -> bool:
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
            data_wellness, float(hrv) if hrv else 0, float(rhr) if rhr else 0,
            float(sono_horas) if sono_horas else 0, int(stress) if stress else 0,
            float(wellness_score) if wellness_score else 0,
            recuperacao_pattern, hrv_guided_suggestion,
            javaloyes_status,
            float(javaloyes_swc.get('inferior', 0) or 0) if javaloyes_swc else 0,
            float(javaloyes_swc.get('superior', 0) or 0) if javaloyes_swc else 0,
            kiviniemi_status,
            float(kiviniemi_swc.get('inferior', 0) or 0) if kiviniemi_swc else 0,
            float(kiviniemi_swc.get('superior', 0) or 0) if kiviniemi_swc else 0,
            float(baseline_lag60) if baseline_lag60 else 0,
            ((float(hrv) - float(baseline_lag60)) / float(baseline_lag60) * 100) 
                if (baseline_lag60 and hrv) else 0,
            notas
        ))
        conn.commit()
        conn.close()
        
        return _upload_hrv_db()
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
            float(hrv_stats.get('media', 0) or 0), float(hrv_stats.get('std', 0) or 0),
            float(hrv_stats.get('min', 0) or 0), float(hrv_stats.get('max', 0) or 0),
            float(rhr_media) if rhr_media else 0,
            float(sono_media) if sono_media else 0,
            float(stress_media) if stress_media else 0,
            padroes_identificados, recomendacoes, notas
        ))
        conn.commit()
        conn.close()
        
        return _upload_hrv_db()
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
