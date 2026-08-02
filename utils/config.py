import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import gspread
from gspread_dataframe import get_as_dataframe
from google.oauth2.service_account import Credentials
from scipy import stats as scipy_stats
from scipy.stats import pearsonr, linregress, spearmanr, theilslopes, kruskal, mannwhitneyu
from itertools import combinations
from datetime import datetime, timedelta
import re
import warnings
import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

# ── URLs e credenciais ────────────────────────────────────────────────────────
WELLNESS_URL  = "https://docs.google.com/spreadsheets/d/10pefcY6VI4Z45M8Y69D6JxIoqOkjzSlSpV1PMLXoYlI/edit#gid=286320937"
FOOD_URL      = "https://docs.google.com/spreadsheets/d/10pefcY6VI4Z45M8Y69D6JxIoqOkjzSlSpV1PMLXoYlI/edit?usp=sharing"
TRAINING_URL  = "https://docs.google.com/spreadsheets/d/1RE4SISd53WmAgQo8J-k2SE_OG0w5m4dbgLHvZHPxKvw/edit?usp=sharing"
# Planilha Annual — AquecSki, AquecBike, AquecRow (igual ao original Python)
ANNUAL_SPREADSHEET_ID = "1AEKhDrda9xhxRQA_1ty3z3oPELzH6oANa6L0cysJSMk"
ANNUAL_SHEETS = ["AquecSki", "AquecBike", "AquecRow"]
SCOPES = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

CORES = {
    'verde': '#2ECC71', 'verde_escuro': '#1D8348',
    'azul': '#3498DB',  'azul_escuro':  '#2471A3',
    'laranja': '#F39C12', 'amarelo': '#F4D03F',
    'vermelho': '#E74C3C', 'vermelho_escuro': '#C0392B',
    'roxo': '#9B59B6', 'cinza': '#7F8C8D',
    'preto': '#2C3E50', 'branco': '#FFFFFF',
}
CORES_ATIV = {
    'Bike': '#E74C3C', 'Run': '#2ECC71', 'Row': '#3498DB',
    'Ski': '#9B59B6',  'WeightTraining': '#F39C12', 'Other': '#7F8C8D',
}

TYPE_MAP = {
    'VirtualSki': 'Ski', 'AlpineSki': 'Ski', 'Ski': 'Ski', 'NordicSki': 'Ski',
    'VirtualRow': 'Row', 'Rowing': 'Row', 'Row': 'Row',
    'VirtualRide': 'Bike', 'Cycling': 'Bike', 'Ride': 'Bike',
    'Bike': 'Bike', 'MountainBike': 'Bike', 'GravelRide': 'Bike',
    'VirtualRun': 'Run', 'Running': 'Run', 'Run': 'Run', 'TrailRun': 'Run', 'Treadmill': 'Run',
    'WeightTraining': 'WeightTraining',
}
VALID_TYPES = ['Bike', 'Row', 'Run', 'Ski', 'WeightTraining']
CICLICOS   = ['Bike', 'Row', 'Run', 'Ski']

MAPA_WELLNESS = {
    'hrv':          ['HRV', 'hrv', 'Heart Rate Variability'],
    'rhr':          ['HRR', 'RHR', 'rhr', 'RestingHR', 'Resting HR'],
    'sleep_hours':  ['Horas de Sono', 'Sleep', 'sleep', 'Hours Sleep'],
    'sleep_quality':['Sono Qualidade', 'Sono_Qualidade', 'Sleep Quality',
                     'Qualidade do Sono', 'Qualidade Sono', 'sleep_quality',
                     'Como foi o seu sono?', 'Sono', 'Quality Sleep'],
    'stress':       ['Stress Do dia', 'Stress', 'stress'],
    'fatiga':       ['Cansaço/Vontade de Treinar', 'Fatiga', 'Fadiga'],
    'humor':        ['Humor', 'humor', 'Mood'],
    'soreness':     ['Cansaço Muscular Geral', 'Muscle Soreness', 'Soreness'],
    'peso':         ['Peso', 'Weight'],
    'fat':          ['FAT', 'Fat', 'Gordura'],
    'hf_power':     ['HF Power', 'HF_Power', 'hf_power', 'HF', 'hf', 'hrv_hf', 'HRV_HF', 'hf power'],
}

MAPA_TRAINING = {
    'date':             ['start_date_local', 'date', 'Date', 'data'],
    'start_date_local': ['start_date_local'],
    'moving_time':      ['moving_time', 'duration', 'Duration'],
    'distance':         ['distance', 'Distance'],
    'power_avg':        ['icu_average_watts', 'average_watts', 'AvgPower'],
    'power_max':        ['MaxPwr', 'max_power', 'Peak5m'],
    'hr_avg':           ['average_heartrate', 'avg_hr'],
    'hr_max':           ['max_heartrate', 'max_hr'],
    'rpe':              ['icu_rpe', 'RPE', 'rpe'],
    'elevation':        ['total_elevation_gain', 'elevation'],
    'type':             ['type', 'sport'],
    'name':             ['name', 'Name'],
    'xss':              ['SS', 'XSS', 'xss', 'strain_score'],
    'pmax':             ['Pmax', 'pmax', 'p_max_usage'],
    'p_max':            ['p_max', 'P_max'],
    'icu_pm_p_max':     ['icu_pm_p_max'],
    'icu_pm_cp':        ['icu_pm_cp', 'cp'],
    'icu_training_load':['icu_training_load'],
    'glycolytic':       ['Glycolytic', 'glycolytic', 'glycolytic_usage'],
    'aerobic':          ['Aerobic', 'aerobic', 'cp_usage'],
    'cadence_avg':      ['average_cadence', 'cadence'],
    'decoupling':       ['CardiacDrift', 'decoupling'],
    'icu_ftp':          ['icu_ftp', 'FTP', 'ftp'],
    'icu_eftp':         ['icu_eftp', 'eFTP', 'estimated_cp', 'est_cp', 'EFTP'],
    'z1_secs': ['z1_secs'], 'z2_secs': ['z2_secs'],
    'z3_secs': ['z3_secs'], 'z4_secs': ['z4_secs'],
    'z5_secs': ['z5_secs'], 'z6_secs': ['z6_secs'],
    'hr_z1_secs': ['hr_z1_secs'], 'hr_z2_secs': ['hr_z2_secs'],
    'hr_z3_secs': ['hr_z3_secs'], 'hr_z4_secs': ['hr_z4_secs'],
    'hr_z5_secs': ['hr_z5_secs'], 'hr_z6_secs': ['hr_z6_secs'],
    'hr_z7_secs': ['hr_z7_secs'],
    'icu_joules': ['icu_joules'], 'icu_weight': ['icu_weight'],
    'AllWorkFTP': ['AllWorkFTP', 'allworkftp', 'Work>FTP', 'WorkAboveFTP'], 'WorkHourKgoverCP': ['WorkHourKgoverCP'],
    'WorkHour':   ['WorkHour'],
    # ── 3-zone power model (desde 2023) ─────────────────────────────────
    'z1_kj':  ['Z1KJ'],
    'z2_kj':  ['Z2KJ'],
    'z3_kj':  ['Z3KJ'],
    'z1_pwr': ['Z1Pw'],
    'z2_pwr': ['Z2pwr'],
    'z3_pwr': ['ZPwr'],
    'z1_sec': ['Z1sec'],
    'z2_sec': ['Z2sec'],
    'z3_sec': ['Z3sec'],
    # ── eW (estimated W') e eFTP por actividade ──────────────────────────
    # icu_pm_w_prime = eW = W' estimado pelo modelo Morton 3P para ESTA sessão
    #   → varia por sessão e modalidade porque o CP/FTP varia
    #   → unidades: JOULES (dividir por 1000 para kJ)
    # icu_w_prime = W' calculado por modalidade (valor fixo de referência)
    #   → referência canónica por desporto, não muda por sessão
    'icu_pm_w_prime':   ['icu_pm_w_prime', 'eW', 'W_prime', 'w_prime'],
    'icu_w_prime':      ['icu_w_prime'],
    # ── HR quartis por actividade (Intervals.icu) ────────────────────────
    # hq_1..4 = média de FC no quartil 1..4 da actividade
    # Nomes exactos na sheet: Hq1, Hq2, Hq3, Hq4
    # hq_4/hq_1 = HR drift ratio = proxy de intensidade intra-sessão
    # Paper FMT 2019: "HR quartiles" são uma das dimensões do vector de estado
    'hq_1': ['Hq1', 'hq_1', 'HQ1', 'hq1'],
    'hq_2': ['Hq2', 'hq_2', 'HQ2', 'hq2'],
    'hq_3': ['Hq3', 'hq_3', 'HQ3', 'hq3'],
    'hq_4': ['Hq4', 'hq_4', 'HQ4', 'hq4'],
    # ── MMP (Mean Maximal Power) por actividade ──────────────────────────
    # Formato: "Yes - 318w"  ou  "No (PR: 364w)"
    # "Yes" = PR na sessão actual  |  "No (PR: Xw)" = PR histórico (season best)
    # Parseados por parse_mmp() em helpers.py — devolvem (is_pr: bool, watts: float)
    # As colunas mmp_*_w contêm apenas os watts; mmp_*_is_pr contém o flag boolean
    'mmp1_raw':  ['MMP1'],
    'mmp3_raw':  ['MMP3'],
    'mmp5_raw':  ['MMP5'],
    'mmp12_raw': ['MMP12'],
    'mmp20_raw': ['MMP20'],
    'mmp60_raw': ['MMP60'],
}

# ── Durações MMP em segundos (para referência e fitting de γ) ────────────────
MMP_DURACOES = {
    'mmp1':  60,      # 1 min
    'mmp3':  180,     # 3 min
    'mmp5':  300,     # 5 min
    'mmp12': 720,     # 12 min
    'mmp20': 1200,    # 20 min
    'mmp60': 3600,    # 60 min
}

# Cores por modalidade — global, disponível em todas as tabs
CORES_MOD = {
    'Bike': CORES['vermelho'],
    'Row':  CORES['azul'],
    'Ski':  CORES['roxo'],
    'Run':  CORES['verde'],
    'WeightTraining': CORES['laranja'],
}
