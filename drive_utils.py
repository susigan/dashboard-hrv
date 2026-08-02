"""
Google Drive Utils — usando googleapiclient (como antes)
Compatível com Streamlit Cloud
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
import io

# ════════════════════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════════════════════

_DRIVE_SCOPES = [
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/drive.file",
]

_FOLDER_NAME = "SQLite"  # Procurar pasta por nome

# ════════════════════════════════════════════════════════════════════════════════
# GOOGLE DRIVE CONNECTION
# ════════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def get_drive_connection():
    """Conecta ao Google Drive via googleapiclient."""
    try:
        creds = Credentials.from_service_account_info(
            dict(st.secrets["gcp_service_account"]), 
            scopes=_DRIVE_SCOPES
        )
        drive = build("drive", "v3", credentials=creds)
        st.success("✅ Google Drive conectado")
        return drive
    except Exception as e:
        st.warning(f"⚠️ Erro Google Drive: {e}")
        return None

# ════════════════════════════════════════════════════════════════════════════════
# FIND FOLDER BY NAME
# ════════════════════════════════════════════════════════════════════════════════

def _find_folder_id(drive, folder_name):
    """Procura pasta por nome no Google Drive."""
    try:
        results = drive.files().list(
            q=f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false",
            fields="files(id)",
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ).execute()
        
        folders = results.get("files", [])
        return folders[0]["id"] if folders else None
    except Exception:
        return None

# ════════════════════════════════════════════════════════════════════════════════
# UPLOAD (para CSV)
# ════════════════════════════════════════════════════════════════════════════════

def upload_resultado_drive(dataframe, filename, folder_name="SQLite"):
    """Upload CSV para Google Drive."""
    try:
        drive = get_drive_connection()
        if drive is None:
            return None
        
        # Procurar pasta
        folder_id = _find_folder_id(drive, folder_name)
        if not folder_id:
            st.error(f"❌ Pasta '{folder_name}' não encontrada")
            return None
        
        # Guardar CSV em memória
        csv_buffer = io.StringIO()
        dataframe.to_csv(csv_buffer, index=False)
        csv_bytes = csv_buffer.getvalue().encode('utf-8')
        
        # Upload
        media = MediaFileUpload(
            io.BytesIO(csv_bytes),
            mimetype="text/csv",
            resumable=True
        )
        
        file_metadata = {
            'name': filename,
            'parents': [folder_id]
        }
        
        file = drive.files().create(
            body=file_metadata,
            media_body=media,
            fields='id'
        ).execute()
        
        st.success(f"✅ Guardado: {filename}")
        return file.get('id')
        
    except Exception as e:
        st.warning(f"⚠️ Erro ao guardar: {e}")
        return None

# ════════════════════════════════════════════════════════════════════════════════
# LIST RESULTS
# ════════════════════════════════════════════════════════════════════════════════

def list_results_drive(folder_name="SQLite"):
    """Lista ficheiros CSV na pasta do Drive."""
    try:
        drive = get_drive_connection()
        if drive is None:
            return []
        
        folder_id = _find_folder_id(drive, folder_name)
        if not folder_id:
            return []
        
        results = drive.files().list(
            q=f"'{folder_id}' in parents and mimeType='text/csv' and trashed=false",
            fields="files(id, name, createdTime)",
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ).execute()
        
        files = results.get("files", [])
        return [
            {
                'id': f['id'],
                'title': f['name'],
                'createdDate': f.get('createdTime', '')
            }
            for f in files
        ]
        
    except Exception as e:
        st.warning(f"⚠️ Erro ao listar: {e}")
        return []

# ════════════════════════════════════════════════════════════════════════════════
# DOWNLOAD
# ════════════════════════════════════════════════════════════════════════════════

def download_resultado_drive(file_id):
    """Download ficheiro do Drive."""
    try:
        drive = get_drive_connection()
        if drive is None:
            return None
        
        # Download para memória
        request = drive.files().get_media(fileId=file_id)
        file_content = request.execute()
        
        # Ler como DataFrame
        csv_buffer = io.BytesIO(file_content)
        df = pd.read_csv(csv_buffer)
        
        st.success(f"✅ Carregado ficheiro")
        return df
        
    except Exception as e:
        st.warning(f"⚠️ Erro ao carregar: {e}")
        return None

# ════════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ════════════════════════════════════════════════════════════════════════════════

__all__ = [
    'get_drive_connection',
    'upload_resultado_drive',
    'list_results_drive',
    'download_resultado_drive'
]
