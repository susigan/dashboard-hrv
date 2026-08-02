"""
ATHELTICA — HRV Analysis & Correlations
CORRIGIDO: Verifica se tabs foram importadas ANTES de chamar
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import sys, os

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# ════════════════════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="ATHELTICA — HRV Analysis",
    page_icon="🔬",
    layout="wide"
)

# ════════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ════════════════════════════════════════════════════════════════════════════════

# Inicializar variáveis
tab_hrv_analyzer = None
tab_correlacoes = None
tabs_available = False

try:
    # Imports da raiz
    from Data_loader import carregar_wellness, carregar_atividades
    from drive_utils import upload_resultado_drive, list_results_drive, download_resultado_drive
    
    # Imports de utils/
    from utils.config import CORES, CORES_ATIV, TYPE_MAP, VALID_TYPES
    from utils.data import preproc_wellness, preproc_ativ
    
    st.success("✅ Imports principais OK")
    
except ImportError as e:
    st.error(f"❌ Erro imports principais: {e}")
    st.stop()

# Tentar importar tabs
try:
    from tabs.tab_hrv_analyzer import tab_hrv_analyzer
    st.success("✅ tab_hrv_analyzer OK")
except ImportError as e:
    st.warning(f"⚠️ tab_hrv_analyzer: {e}")
    tab_hrv_analyzer = None

try:
    from tabs.tab_correlacoes import tab_correlacoes
    st.success("✅ tab_correlacoes OK")
except ImportError as e:
    st.warning(f"⚠️ tab_correlacoes: {e}")
    tab_correlacoes = None

# Verificar se ambas foram carregadas
if tab_hrv_analyzer is not None and tab_correlacoes is not None:
    tabs_available = True
    st.success("✅ Ambas as tabs carregadas!")

# ════════════════════════════════════════════════════════════════════════════════
# CARREGAR DADOS
# ════════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=7200)
def load_data():
    try:
        wc = carregar_wellness(9999)
        ac = carregar_atividades(9999)
        return wc, ac
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return None, None

wc, ac = load_data()

if wc is None or ac is None:
    st.stop()

# ════════════════════════════════════════════════════════════════════════════════
# TITULO
# ════════════════════════════════════════════════════════════════════════════════

st.title("🔬 ATHELTICA — HRV Analysis")

# ════════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════════

st.sidebar.title("🔬 HRV Analysis")
st.sidebar.info("App dedicada para análises HRV")

st.sidebar.markdown("---")
st.sidebar.subheader("💾 Google Drive Storage")

with st.sidebar.expander("📂 Histórico", expanded=False):
    try:
        results = list_results_drive(folder_name="SQLite")
        if results:
            st.write(f"✅ {len(results)} resultados")
        else:
            st.info("📭 Sem resultados ainda")
    except Exception:
        st.warning("⚠️ Drive não disponível")

# ════════════════════════════════════════════════════════════════════════════════
# CONTEÚDO PRINCIPAL
# ════════════════════════════════════════════════════════════════════════════════

st.markdown("---")

if not tabs_available:
    st.error("❌ Tabs não puderam ser carregadas. Verifica os erros acima!")
    st.stop()

st.success("✅ Renderizando tabs...")

tabs = st.tabs(["🔬 Recovery Patterns", "🧠 Correlações"])

# ─ TAB 1: tab_hrv_analyzer ─────────────────────────────────────────────────────
with tabs[0]:
    if tab_hrv_analyzer is None:
        st.error("❌ tab_hrv_analyzer não está disponível")
    else:
        try:
            tab_hrv_analyzer(wc, ac, wc_full=wc, da_full=ac)
        except Exception as e:
            st.error(f"❌ Erro executando tab_hrv_analyzer: {e}")
            import traceback
            with st.expander("📋 Traceback"):
                st.code(traceback.format_exc())

# ─ TAB 2: tab_correlacoes ──────────────────────────────────────────────────────
with tabs[1]:
    if tab_correlacoes is None:
        st.error("❌ tab_correlacoes não está disponível")
    else:
        try:
            tab_correlacoes(ac, wc)
        except Exception as e:
            st.error(f"❌ Erro executando tab_correlacoes: {e}")
            import traceback
            with st.expander("📋 Traceback"):
                st.code(traceback.format_exc())

# ════════════════════════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.caption(f"ATHELTICA HRV | {len(wc)} wellness | {len(ac)} atividades")
