# Copyright (c) 2026 Ventilator Lab AI
# Propiedad Intelectual Privada - Juan Martín Nuñez Silveira
# Desarrollado para fines educativos y soporte clínico en Ventilación Mecánica.

import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
import matplotlib
from PIL import Image
import io
import google.generativeai as genai
import os

# Configuración de Matplotlib para estética oscura (UCI Mode)
matplotlib.use('Agg')
plt.style.use('dark_background')

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Ventilator Lab | AI Diagnostics",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Estilos CSS Personalizados ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1f2937; padding: 15px; border-radius: 10px; border: 1px solid #374151; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #2563eb; color: white; border: none; font-weight: bold; }
    .stButton>button:hover { background-color: #1d4ed8; border: none; }
    .stCamera>div>div>div { border-radius: 20px; border: 2px solid #3b82f6; }
    h1 { color: #f3f4f6; font-family: 'Inter', sans-serif; font-weight: 700; }
    h3 { color: #9ca3af; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 0. DICCIONARIO DE TEXTOS Y PROMPTS
# ==========================================
TEXTOS = {
    "es": {
        "title": "VENTILATOR LAB AI",
        "tagline": "Clinical Decision Support System | Especialidad Cuidados Intensivos",
        "col_config": "🛠️ CONFIGURACIÓN TÉCNICA",
        "col_tools": "🔬 PANEL DE ANÁLISIS",
        "api_key_label": "Clave de Acceso (Google AI Studio)",
        "license_active": "✅ CONEXIÓN ESTABLECIDA",
        "select_curve": "SELECCIONAR CANAL DE MONITOREO",
        "opt_p": "Presión de Vía Aérea (Paw)",
        "opt_f": "Flujo Inspiratorio/Espiratorio",
        "header_calib": "CALIBRACIÓN ÓPTICA",
        "camera_msg": "Capturar pantalla del ventilador",
        "btn_ai": "EJECUTAR AUDITORÍA CLÍNICA IA",
        "ai_report_title": "📝 REPORTE DE CONSULTORÍA IA",
        "math_result": "Análisis Geométrico Automatizado",
        "prompt_instructions": """
        Actúa como un Auditor Clínico Senior especialista en Ventilación Mecánica.
        Analiza la imagen de la curva de **{tipo_curva}**:
        1. MORFOLOGÍA: Describe la onda observada.
        2. HALLAZGOS: Identifica asincronías (Doble disparo, Hambre de flujo, Ciclado prematuro, Esfuerzos inefectivos).
        3. ACCIÓN: Sugerencia técnica breve para el operador.
        Responde en español con formato Markdown profesional.
        """
    }
}

# ==========================================
# 1. MOTOR DE IA (CON SOLUCIÓN ERROR 404)
# ==========================================

def get_ai_response(image_bytes, tipo_curva, api_key):
    try:
        genai.configure(api_key=api_key)
        img = Image.open(io.BytesIO(image_bytes))
        img.thumbnail((1000, 1000))
        
        # --- Solución dinámica de Modelos ---
        modelos_visibles = genai.list_models()
        available_names = [m.name for m in modelos_visibles if 'generateContent' in m.supported_generation_methods]
        
        # Prioridad de modelos para Visión
        prioridad = [
            "models/gemini-1.5-flash", 
            "models/gemini-1.5-flash-latest", 
            "models/gemini-1.5-pro", 
            "models/gemini-pro-vision"
        ]
        
        modelo_final = next((m for m in prioridad if m in available_names), None)
        
        if not modelo_final:
            if len(available_names) > 0:
                modelo_final = available_names[0]
            else:
                return "❌ No se encontraron modelos compatibles con Vision en esta API Key."

        model = genai.GenerativeModel(modelo_final)
        prompt = TEXTOS["es"]["prompt_instructions"].format(tipo_curva=tipo_curva)
        
        response = model.generate_content([prompt, img], generation_config={"temperature": 0.1})
        return response.text
        
    except Exception as e:
        return f"❌ Error de configuración: {str(e)}"

# ==========================================
# 2. INTERFAZ DE USUARIO (DASHBOARD)
# ==========================================

def main():
    t = TEXTOS["es"]
    
    # Encabezado
    st.title(t["title"])
    st.write(t["tagline"])
    st.divider()

    # Layout Principal
    col_left, col_right = st.columns([1, 2], gap="large")

    with col_left:
        st.subheader(t["col_config"])
        
        # Gestión de API Key
        api_key = st.secrets.get("GOOGLE_API_KEY") or st.text_input(t["api_key_label"], type="password")
        if api_key:
            st.success(t["license_active"])
        
        st.divider()
        
        # Selección de canal
        canal = st.selectbox(t["select_curve"], [t["opt_p"], t["opt_f"]])
        tipo_logic = "pressure" if t["opt_p"] in canal else "flow"
        
        # Calibración Óptica
        with st.expander(t["header_calib"], expanded=True):
            def_h = (20, 45) if tipo_logic == "pressure" else (85, 110)
            h_min, h_max = st.slider("Matiz (Hue)", 0, 179, def_h)
            s_val = st.slider("Saturación", 0, 255, (100, 255))
            v_val = st.slider("Brillo (Value)", 0, 255, (100, 255))

    with col_right:
        st.subheader(t["col_tools"])
        
        # Entrada de Cámara
        input_img = st.camera_input(t["camera_msg"])

        if input_img:
            bytes_data = input_img.getvalue()
            img_cv = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
            
            mask = cv2.inRange(hsv, np.array([h_min, s_val[0], v_val[0]]), np.array([h_max, s_val[1], v_val[1]]))
            
            # --- Extracción de Curva y Limpieza de Ruido ---
            h_px, w_px = mask.shape
            signal = []
            last_valid = 0
            for col in range(int(w_px*0.1), int(w_px*0.9)):
                col_data = mask[:, col]
                if np.max(col_data) > 0:
                    val = h_px - np.argmax(col_data)
                    # Filtro de continuidad para evitar saltos por reflejos
                    if len(signal) > 0 and abs(val - last_valid) > (h_px * 0.3):
                        signal.append(last_valid)
                    else:
                        signal.append(val)
                        last_valid = val
                else:
                    signal.append(last_valid)
            
            # --- Visualización del Monitor (Look Hamilton/Dräger) ---
            if np.max(signal) > 0:
                sig_norm = (np.array(signal) - np.min(signal)) / (np.max(signal) - np.min(signal) + 1e-6)
                sig_smooth = savgol_filter(sig_norm, 31, 3)
                
                fig, ax = plt.subplots(figsize=(10, 3))
                color_wave = "#fbbf24" if tipo_logic == "pressure" else "#22d3ee"
                ax.plot(sig_smooth, color=color_wave, lw=2.5)
                ax.fill_between(range(len(sig_smooth)), sig_smooth, color=color_wave, alpha=0.15)
                ax.grid(color='#374151', linestyle='--', linewidth=0.5, alpha=0.5)
                ax.axis('off')
                st.pyplot(fig)
                
                st.metric(t["math_result"], "SEÑAL CAPTURADA", delta="OK")
            else:
                st.warning("⚠️ No se detecta señal. Ajusta la 'Calibración Óptica' en el panel izquierdo.")
            
            st.divider()
            
            # Acción IA
            if st.button(t["btn_ai"]):
                if not api_key:
                    st.error("Por favor, ingresa una API Key válida.")
                else:
                    with st.status("🧠 Iniciando consultoría diagnóstica..."):
                        reporte = get_ai_response(bytes_data, canal, api_key)
                    
                    st.subheader(t["ai_report_title"])
                    st.markdown(reporte)

if __name__ == "__main__":
    main()
