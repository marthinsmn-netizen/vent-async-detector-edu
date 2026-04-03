# Copyright (c) 2026 Ventilator Lab AI
# Propiedad Intelectual Privada - Juan Martín Nuñez Silveira
# Versión Optimizada para Producción - Independencia Financiera & Contenido de Impacto

import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
import matplotlib
from PIL import Image
import io
import google.generativeai as genai
import time
from datetime import datetime

# Intentamos importar el generador de PDF, si no existe, creamos un dummy para que no falle
try:
    from pdf_report import boton_descarga_pdf
except ImportError:
    def boton_descarga_pdf(*args, **kwargs):
        pass

# --- Configuración Técnica ---
matplotlib.use('Agg')
plt.style.use('dark_background')

st.set_page_config(
    page_title="Ventilator Lab | AI Diagnostics",
    page_icon="🫁",
    layout="wide"
)

# --- Estilos CSS (UCI Dark Mode) ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1f2937; padding: 15px; border-radius: 10px; border: 1px solid #374151; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #2563eb; color: white; font-weight: bold; }
    h1 { color: #f3f4f6; font-family: 'Inter', sans-serif; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 1. MOTOR MATEMÁTICO (DETERMINISTA)
# ==========================================

def analizar_senal_matematica(signal, tipo_logic):
    hallazgos = {"doble_disparo": False, "hambre_flujo": False, "picos": 0, "fr": 0, "resumen": ""}
    
    if len(signal) < 10:
        return hallazgos

    sig = np.array(signal, dtype=float)
    sig_norm = (sig - np.min(sig)) / (np.max(sig) - np.min(sig) + 1e-6)
    
    # Suavizado Savitzky-Golay
    win = min(31, len(sig_norm) - 1)
    if win % 2 == 0: win -= 1
    sig_smooth = savgol_filter(sig_norm, win, 3) if win >= 5 else sig_norm

    peaks, _ = find_peaks(sig_smooth, prominence=0.15, distance=10)
    hallazgos["picos"] = len(peaks)

    # Lógica de Doble Disparo (Simplificada para eficiencia)
    if len(peaks) >= 2:
        distancias = np.diff(peaks)
        ciclo_medio = np.mean(distancias)
        hallazgos["fr"] = round((len(peaks) / 10) * 60, 1)
        
        for i in range(len(peaks) - 1):
            if (peaks[i+1] - peaks[i]) < ciclo_medio * 0.5:
                hallazgos["doble_disparo"] = True
                break

    # Lógica Hambre de Flujo (Muesca en Presión)
    if tipo_logic == "pressure" and len(peaks) > 0:
        d2 = np.diff(np.diff(sig_smooth))
        if np.sum(d2 < -0.01) > len(d2) * 0.3:
            hallazgos["hambre_flujo"] = True

    hallazgos["resumen"] = f"Picos:{hallazgos['picos']}|FR:{hallazgos['fr']}|DD:{hallazgos['doble_disparo']}|HF:{hallazgos['hambre_flujo']}"
    return hallazgos

# ==========================================
# 2. MOTOR DE IA (OPTIMIZADO)
# ==========================================

def get_ai_response(image_bytes, tipo_curva, api_key, contexto, pre_diag, signal):
    genai.configure(api_key=api_key)
    
    # Reducción de resolución de imagen para ahorrar tokens de entrada
    img = Image.open(io.BytesIO(image_bytes))
    img.thumbnail((800, 800))

    # Downsampling de la señal (1 cada 10 puntos) para el prompt
    signal_summary = ",".join([str(round(v, 1)) for v in signal[::10]])

    # Prompt Ultra-Optimizado (Bajo consumo de tokens)
    prompt = f"""
    Rol: Auditor UCI Senior. 
    Contexto: Modo {contexto['modo_vent']}, PEEP {contexto['peep']}, FiO2 {contexto['fio2']}.
    Curva: {tipo_curva}. 
    Pre-diag Matemático: {pre_diag}.
    Datos Crudos (resumen): {signal_summary}
    
    Analiza la imagen y datos:
    1. Morfología: Breve.
    2. Asincronías: Identificar (DD, HF, Ciclado, Esfuerzo Inef).
    3. Acción: Sugerencia técnica inmediata.
    Formato: Markdown corto. Evita introducciones.
    """

    # Fallback de modelos (Prioridad a los más económicos/rápidos)
    modelos = ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-flash-8b"]
    
    for mod_name in modelos:
        try:
            model = genai.GenerativeModel(mod_name)
            # Manejo de reintentos interno
            for attempt in range(2):
                try:
                    response = model.generate_content([prompt, img], generation_config={"temperature": 0.1, "max_output_tokens": 300})
                    return response.text, mod_name
                except Exception as e:
                    if "429" in str(e):
                        time.sleep(5) # Espera corta antes de reintentar o saltar
                        continue
                    raise e
        except Exception:
            continue # Probar el siguiente modelo si este falla
            
    return "⚠️ Límite de cuota excedido. Por favor, intenta en 1 minuto.", "N/A"

# ==========================================
# 3. INTERFAZ (STREAMLIT)
# ==========================================

def main():
    if "historial" not in st.session_state: st.session_state.historial = []

    st.title("🫁 VENTILATOR LAB AI")
    st.caption("Clinical Decision Support System | Especialidad Cuidados Intensivos")
    
    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.subheader("🛠️ CONFIG")
        # Prioridad a Secrets de Streamlit, luego input manual
        api_key = st.secrets.get("GOOGLE_API_KEY") or st.text_input("API Key (Google AI Studio)", type="password")
        
        canal = st.selectbox("Canal de Monitoreo", ["Presión (Paw)", "Flujo (Flow)"])
        tipo_logic = "pressure" if "Presión" in canal else "flow"
        
        with st.expander("🏥 CONTEXTO CLÍNICO", expanded=True):
            modo = st.selectbox("Modo", ["VCV", "PCV", "PSV", "SIMV", "Otros"])
            fio2 = st.slider("FiO2", 21, 100, 40)
            peep = st.slider("PEEP", 0, 25, 5)
            obs = st.text_area("Notas")

    with col_right:
        tab1, tab2 = st.tabs(["📊 Análisis", "📋 Historial"])
        
        with tab1:
            input_mode = st.radio("Fuente", ["📷 Cámara", "📁 Archivo"], horizontal=True)
            input_img = st.camera_input("Captura") if input_mode == "📷 Cámara" else st.file_uploader("Subir", type=['jpg','png'])

            if input_img:
                bytes_data = input_img.getvalue()
                img_cv = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                
                # --- Procesamiento Óptico Simple ---
                hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
                # Máscara adaptada a colores comunes de monitores (amarillo/cian)
                mask = cv2.inRange(hsv, np.array([20, 100, 100]), np.array([110, 255, 255]))
                
                signal = []
                h_px, w_px = mask.shape
                for col in range(int(w_px * 0.1), int(w_px * 0.9), 2): # Step 2 para velocidad
                    col_data = mask[:, col]
                    if np.max(col_data) > 0:
                        signal.append(h_px - np.argmax(col_data))
                
                if signal:
                    # Gráfico
                    fig, ax = plt.subplots(figsize=(8, 2))
                    ax.plot(signal, color="#fbbf24" if tipo_logic=="pressure" else "#22d3ee")
                    ax.axis('off')
                    st.pyplot(fig)
                    
                    hallazgos = analizar_senal_matematica(signal, tipo_logic)
                    st.write(f"**Motor Matemático:** {hallazgos['resumen']}")
                    
                    if st.button("🧠 AUDITORÍA IA"):
                        if not api_key:
                            st.error("Falta API KEY")
                        else:
                            with st.spinner("Analizando..."):
                                contexto = {"modo_vent": modo, "fio2": fio2, "peep": peep}
                                reporte, mod = get_ai_response(bytes_data, canal, api_key, contexto, hallazgos['resumen'], signal)
                                st.markdown(reporte)
                                st.session_state.historial.append({"t": datetime.now().strftime("%H:%M"), "rep": reporte})

        with tab2:
            for item in reversed(st.session_state.historial):
                st.write(f"**{item['t']}**")
                st.markdown(item['rep'])
                st.divider()

if __name__ == "__main__":
    main()
