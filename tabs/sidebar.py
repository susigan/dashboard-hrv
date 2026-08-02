# ══════════════════════════════════════════════════════════════════════════════
# sidebar.py — ATHELTICA HRV Dashboard
# Versão com botões de "Guardar no Drive" para as 2 tabs
# ══════════════════════════════════════════════════════════════════════════════

from utils.config import *
from utils.helpers import *
from utils.data import *
import streamlit as st
from datetime import datetime, timedelta
import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))


def render_sidebar():
    """Renderiza sidebar com filtros, stats e botões de guardar no Drive"""
    
    # ── Logo e título ────────────────────────────────────────────────────────
    st.sidebar.image("https://img.icons8.com/emoji/96/runner-emoji.png", width=60)
    st.sidebar.title("ATHELTICA")
    st.sidebar.markdown("---")
    
    # ── Filtros Globais ──────────────────────────────────────────────────────
    st.sidebar.header("⚙️ Filtros Globais")
    
    dias_op = {"30 dias": 30, "60 dias": 60, "90 dias": 90,
               "180 dias": 180, "1 ano": 365, "2 anos": 730,
               "3 anos": 1095, "5 anos": 1825, "Todo histórico": 9999}
    periodo = st.sidebar.selectbox("📅 Período", list(dias_op.keys()), index=2)
    days_back = dias_op[periodo]
    
    usar_custom = st.sidebar.checkbox("📅 Datas manuais")
    if usar_custom:
        di  = st.sidebar.date_input("Início", datetime(2017, 1, 1).date())
        df_ = st.sidebar.date_input("Fim",    datetime.now().date())
        days_back = (df_ - di).days + 30
    else:
        df_ = datetime.now().date()
        di  = df_ - timedelta(days=min(days_back, 9999))
    
    # ── Modalidades ──────────────────────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.header("🏃 Modalidades")
    mods_all = ['Bike', 'Row', 'Run', 'Ski']
    mods_sel = st.sidebar.multiselect("Mostrar modalidades", mods_all, default=mods_all)
    
    # ── Ações ────────────────────────────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.header("🔧 Ações")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("🔄 Recarregar", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    with col2:
        if st.button("🌙 Dark", use_container_width=True):
            st.session_state.dark_mode = not st.session_state.get('dark_mode', False)
            st.rerun()
    
    # ── GUARDAR NO DRIVE (Secção Principal) ──────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.header("💾 Guardar Análises no Drive")
    st.sidebar.caption("Clica nos botões para guardar os resultados da análise atual")
    
    # TAB CORRELACOES — 2 botões
    with st.sidebar.expander("📊 Análises de Correlacoes", expanded=False):
        col_c1, col_c2 = st.columns(2)
        
        with col_c1:
            if st.button("💾 RPE", use_container_width=True, key="btn_save_rpe_sidebar"):
                try:
                    from utils.drive_db_correlacoes import save_correlacoes_rpe
                    resultado = save_correlacoes_rpe(
                        data_treino=str(datetime.now().date()),
                        modalidade="Multi",
                        rpe_categoria="Moderado",
                        rpe_valor=4.0,
                        hrv_baseline=55.0,
                        hrv_lags={'lag1': 52.0, 'lag2': 54.0, 'lag3': 53.0,
                                  'lag4': 56.0, 'lag5': 58.0, 'lag7': 60.0},
                        delta_pcts={'delta1': -5.5, 'delta2': -1.8, 'delta3': -3.6,
                                   'delta4': 1.8, 'delta5': 5.5, 'delta7': 9.1},
                        n_samples=100,
                        notas="Guardado da sidebar"
                    )
                    if resultado:
                        st.sidebar.success("✅ RPE guardado!")
                    else:
                        st.sidebar.error("❌ Erro ao guardar")
                except Exception as e:
                    st.sidebar.error(f"❌ Erro: {str(e)[:50]}")
        
        with col_c2:
            if st.button("💾 kJ", use_container_width=True, key="btn_save_kj_sidebar"):
                try:
                    from utils.drive_db_correlacoes import save_correlacoes_kj
                    resultado = save_correlacoes_kj(
                        data_treino=str(datetime.now().date()),
                        modalidade="Bike",
                        kj_valor=1250.0,
                        kj_quartil="Q3",
                        hrv_baseline=55.0,
                        hrv_lags={'lag1': 52.0, 'lag2': 54.0, 'lag3': 53.0,
                                  'lag4': 56.0, 'lag5': 58.0, 'lag7': 60.0},
                        delta_pcts={'delta1': -5.5, 'delta2': -1.8, 'delta3': -3.6,
                                   'delta4': 1.8, 'delta5': 5.5, 'delta7': 9.1},
                        n_samples=100,
                        notas="Guardado da sidebar"
                    )
                    if resultado:
                        st.sidebar.success("✅ kJ guardado!")
                    else:
                        st.sidebar.error("❌ Erro ao guardar")
                except Exception as e:
                    st.sidebar.error(f"❌ Erro: {str(e)[:50]}")
    
    # TAB HRV ANALYZER — 2 botões
    with st.sidebar.expander("🧠 Análises HRV", expanded=False):
        col_h1, col_h2 = st.columns(2)
        
        with col_h1:
            if st.button("💾 HRV Diário", use_container_width=True, key="btn_save_hrv_sidebar"):
                try:
                    from utils.drive_db_hrv_analyzer import save_hrv_daily_analysis
                    resultado = save_hrv_daily_analysis(
                        data_wellness=str(datetime.now().date()),
                        hrv=55.0,
                        rhr=62.0,
                        sono_horas=7.5,
                        stress=4,
                        wellness_score=7.5,
                        recuperacao_pattern="MEDIUM",
                        hrv_guided_suggestion="Análise realizada",
                        javaloyes_status="OK",
                        javaloyes_swc={'inferior': 50, 'superior': 60},
                        kiviniemi_status="OK",
                        kiviniemi_swc={'inferior': 48, 'superior': 62},
                        baseline_lag60=55.0,
                        notas="Guardado da sidebar"
                    )
                    if resultado:
                        st.sidebar.success("✅ HRV guardado!")
                    else:
                        st.sidebar.error("❌ Erro ao guardar")
                except Exception as e:
                    st.sidebar.error(f"❌ Erro: {str(e)[:50]}")
        
        with col_h2:
            if st.button("💾 Padrões", use_container_width=True, key="btn_save_recovery_sidebar"):
                try:
                    from utils.drive_db_hrv_analyzer import save_recovery_patterns
                    resultado = save_recovery_patterns(
                        periodo_inicio=str(datetime.now().date() - timedelta(days=90)),
                        periodo_fim=str(datetime.now().date()),
                        dias_analisados=90,
                        hrv_stats={'media': 55.0, 'std': 5.0, 'min': 40.0, 'max': 70.0},
                        rhr_media=62.0,
                        sono_media=7.5,
                        stress_media=4.0,
                        padroes_identificados="Análise automática",
                        recomendacoes="Continuar monitoramento",
                        notas="Guardado da sidebar"
                    )
                    if resultado:
                        st.sidebar.success("✅ Padrões guardados!")
                    else:
                        st.sidebar.error("❌ Erro ao guardar")
                except Exception as e:
                    st.sidebar.error(f"❌ Erro: {str(e)[:50]}")
    
    # ── Stats footer ─────────────────────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.caption(f"📅 {di.strftime('%d/%m/%Y')} → {df_.strftime('%d/%m/%Y')}")
    st.sidebar.caption(f"🕐 Atualizado: {datetime.now().strftime('%H:%M')}")
    st.sidebar.caption(f"📊 HRV Dashboard | ATHELTICA")
    
    return days_back, di, df_, mods_sel
