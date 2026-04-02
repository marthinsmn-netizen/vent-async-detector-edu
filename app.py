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
import json
from datetime import datetime

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
    h1 { color: #f3f4f6; font-family: 'Inter', sans-serif; font-weight: 700; }
    h3 { color: #9ca3af; }
    .hallazgo-card { background-color: #1f2937; border-radius: 8px; padding: 12px; margin: 6px 0; border-left: 4px solid #ef4444; }
    .hallazgo-ok { border-left-color: #22c55e; }
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
        "header_calib": "⚙️ CALIBRACIÓN ÓPTICA",
        "header_context": "🏥 CONTEXTO CLÍNICO",
        "camera_msg": "Capturar pantalla del ventilador",
        "upload_msg": "O subir imagen desde archivo",
        "btn_ai": "🧠 EJECUTAR AUDITORÍA CLÍNICA IA",
        "ai_report_title": "📝 REPORTE DE CONSULTORÍA IA",
        "math_result": "Motor Matemático",
        "tab_analysis": "📊 Análisis en Vivo",
        "tab_history": "📋 Historial de Sesión",
        # Prompt enriquecido con contexto clínico
        "prompt_instructions": """
        Actúa como un Auditor Clínico Senior especialista en Ventilación Mecánica Invasiva.
        
        CONTEXTO DEL PACIENTE Y VENTILADOR:
        - Modo ventilatorio: {modo_vent}
        - FiO₂: {fio2}%
        - PEEP: {peep} cmH₂O
        - FR programada: {fr} rpm
        - Observaciones del operador: {obs}
        
        PRE-DIAGNÓSTICO MATEMÁTICO (Motor OpenCV/SciPy):
        {pre_diagnostico}
        
        Analiza la imagen de la curva de **{tipo_curva}** considerando el contexto anterior:
        
        ## 1. MORFOLOGÍA
        Describe la onda observada: forma, amplitud, simetría.
        
        ## 2. HALLAZGOS CLÍNICOS
        Identifica y clasifica cualquier asincronía presente:
        - Doble Disparo (Double Trigger)
        - Hambre de Flujo (Flow Starvation)
        - Ciclado Prematuro / Tardío
        - Esfuerzos Inefectivos
        - Auto-PEEP
        
        ## 3. CORRELACIÓN CON CONTEXTO
        Relaciona los hallazgos con el modo ventilatorio y parámetros reportados.
        
        ## 4. ACCIÓN RECOMENDADA
        Sugerencia técnica concreta y priorizada para el operador.
        
        ## ⚠️ AVISO EDUCATIVO
        Este análisis es de apoyo educativo. No reemplaza el juicio clínico profesional.
        
        Responde en español con formato Markdown profesional.
        """
    }
}

# ==========================================
# 1. MOTOR MATEMÁTICO (CAPA DETERMINISTA)
# ==========================================

def analizar_senal_matematica(signal, tipo_logic):
    """
    Motor determinista híbrido basado en SciPy find_peaks.
    Implementa las reglas heurísticas descritas en el README.
    Retorna: dict con hallazgos y string de resumen para el prompt.
    """
    hallazgos = {
        "doble_disparo": False,
        "hambre_flujo": False,
        "picos_detectados": 0,
        "frecuencia_estimada": 0,
        "amplitud_media": 0,
        "resumen": "Sin datos suficientes para análisis matemático."
    }

    if len(signal) < 10 or np.max(signal) == 0:
        return hallazgos

    sig = np.array(signal, dtype=float)
    sig_norm = (sig - np.min(sig)) / (np.max(sig) - np.min(sig) + 1e-6)

    # Aplicar Savitzky-Golay con ventana adaptativa (fix bug #4)
    win = min(31, len(sig_norm) - 1)
    if win % 2 == 0:
        win -= 1
    if win >= 5:
        sig_smooth = savgol_filter(sig_norm, win, 3)
    else:
        sig_smooth = sig_norm

    # --- Detección de picos ---
    peaks, props = find_peaks(sig_smooth, prominence=0.15, distance=10)
    hallazgos["picos_detectados"] = len(peaks)

    # Estimación de frecuencia respiratoria (asumiendo 80px = 1 ciclo aprox.)
    if len(peaks) >= 2:
        distancias = np.diff(peaks)
        ciclo_medio_px = np.mean(distancias)
        # Asumimos que la imagen representa ~10 segundos de ventana
        segundos_totales = 10
        freq_estimada = (len(peaks) / segundos_totales) * 60
        hallazgos["frecuencia_estimada"] = round(freq_estimada, 1)

    # --- REGLA 1: Doble Disparo ---
    # Dos picos consecutivos con distancia < 1/3 del ciclo medio y valle poco profundo
    if len(peaks) >= 2:
        for i in range(len(peaks) - 1):
            dist = peaks[i+1] - peaks[i]
            if len(peaks) >= 3:
                ciclo_ref = np.mean(np.diff(peaks))
            else:
                ciclo_ref = dist * 2
            if dist < ciclo_ref * 0.6:
                # Verificar que el valle entre picos no sea muy profundo (< 40% de bajada)
                valle_region = sig_smooth[peaks[i]:peaks[i+1]]
                bajada = np.min(valle_region)
                if bajada > sig_smooth[peaks[i]] * 0.5:
                    hallazgos["doble_disparo"] = True
                    break

    # --- REGLA 2: Hambre de Flujo (solo para curva de presión) ---
    # Concavidad anómala = muesca en la rama inspiratoria (derivada segunda negativa sostenida)
    if tipo_logic == "pressure" and len(peaks) > 0:
        for pk in peaks:
            inicio = max(0, pk - 20)
            rama_insp = sig_smooth[inicio:pk]
            if len(rama_insp) > 5:
                d2 = np.diff(np.diff(rama_insp))
                # Si hay una zona de curvatura negativa pronunciada en la subida = muesca
                if np.sum(d2 < -0.01) > len(d2) * 0.4:
                    hallazgos["hambre_flujo"] = True
                    break

    # --- Amplitud media ---
    if len(peaks) > 0:
        hallazgos["amplitud_media"] = round(float(np.mean(sig_smooth[peaks])), 3)

    # --- Construir resumen para el prompt ---
    partes = [f"Picos detectados: {hallazgos['picos_detectados']}"]
    if hallazgos["frecuencia_estimada"] > 0:
        partes.append(f"FR estimada: ~{hallazgos['frecuencia_estimada']} rpm")
    if hallazgos["doble_disparo"]:
        partes.append("⚠️ SOSPECHA DE DOBLE DISPARO detectada por análisis geométrico.")
    if hallazgos["hambre_flujo"]:
        partes.append("⚠️ POSIBLE HAMBRE DE FLUJO detectada (muesca inspiratoria).")
    if not hallazgos["doble_disparo"] and not hallazgos["hambre_flujo"]:
        partes.append("No se detectaron asincronías obvias por análisis matemático.")

    hallazgos["resumen"] = " | ".join(partes)
    return hallazgos


# ==========================================
# 2. MOTOR DE IA (CAPA GENERATIVA)
# ==========================================

def get_ai_response(image_bytes, tipo_curva, api_key, contexto_clinico, pre_diagnostico):
    """Llama a Gemini con contexto clínico enriquecido y pre-diagnóstico matemático."""
    try:
        genai.configure(api_key=api_key)
        img = Image.open(io.BytesIO(image_bytes))
        img.thumbnail((1000, 1000))

        # --- Selección dinámica de modelo ---
        modelos_visibles = genai.list_models()
        available_names = [
            m.name for m in modelos_visibles
            if 'generateContent' in m.supported_generation_methods
        ]

        # Lista de prioridad actualizada (gemini-pro-vision fue deprecado)
        prioridad = [
            "models/gemini-2.0-flash",
            "models/gemini-2.0-flash-lite",
            "models/gemini-1.5-flash",
            "models/gemini-1.5-flash-latest",
            "models/gemini-1.5-pro",
        ]

        modelo_final = next((m for m in prioridad if m in available_names), None)
        if not modelo_final:
            if available_names:
                modelo_final = available_names[0]
            else:
                return "❌ No se encontraron modelos compatibles con visión en esta API Key."

        model = genai.GenerativeModel(modelo_final)

        prompt = TEXTOS["es"]["prompt_instructions"].format(
            tipo_curva=tipo_curva,
            modo_vent=contexto_clinico.get("modo_vent", "No especificado"),
            fio2=contexto_clinico.get("fio2", "N/A"),
            peep=contexto_clinico.get("peep", "N/A"),
            fr=contexto_clinico.get("fr", "N/A"),
            obs=contexto_clinico.get("obs", "Sin observaciones adicionales."),
            pre_diagnostico=pre_diagnostico
        )

        response = model.generate_content(
            [prompt, img],
            generation_config={"temperature": 0.1}
        )

        # Validar respuesta no vacía
        texto = response.text if response.text else "⚠️ La IA no devolvió contenido. Intentá nuevamente."
        return texto, modelo_final

    except Exception as e:
        return f"❌ Error de configuración: {str(e)}", "N/A"


# ==========================================
# 3. INTERFAZ DE USUARIO (DASHBOARD)
# ==========================================

def main():
    t = TEXTOS["es"]

    # Inicializar historial en session_state (fix scope + persistencia de sesión)
    if "historial" not in st.session_state:
        st.session_state.historial = []

    # --- Encabezado ---
    st.title(t["title"])
    st.caption(t["tagline"])
    st.divider()

    # --- Layout Principal ---
    col_left, col_right = st.columns([1, 2], gap="large")

    # ========================
    # COLUMNA IZQUIERDA: Config
    # ========================
    with col_left:
        st.subheader(t["col_config"])

        # --- API Key (con manejo robusto de secrets) ---
        api_key = None
        try:
            api_key = st.secrets.get("GOOGLE_API_KEY")
        except Exception:
            pass
        if not api_key:
            api_key = st.text_input(t["api_key_label"], type="password")
        if api_key:
            st.success(t["license_active"])

        st.divider()

        # --- Selección de canal ---
        canal = st.selectbox(t["select_curve"], [t["opt_p"], t["opt_f"]])
        tipo_logic = "pressure" if t["opt_p"] in canal else "flow"

        # --- Contexto Clínico (nuevo) ---
        with st.expander(t["header_context"], expanded=False):
            modos_vent = ["VCV (Volumen Control)", "PCV (Presión Control)", "PSV (Presión Soporte)",
                          "SIMV", "APRV", "No especificado"]
            modo_vent = st.selectbox("Modo Ventilatorio", modos_vent, index=5)
            col_a, col_b = st.columns(2)
            with col_a:
                fio2 = st.number_input("FiO₂ (%)", 21, 100, 40, step=5)
                peep = st.number_input("PEEP (cmH₂O)", 0, 25, 5)
            with col_b:
                fr = st.number_input("FR prog. (rpm)", 0, 40, 14)
            obs = st.text_area("Observaciones del operador", placeholder="Ej: paciente agitado, secreciones...", height=80)

        contexto_clinico = {
            "modo_vent": modo_vent,
            "fio2": fio2,
            "peep": peep,
            "fr": fr,
            "obs": obs if obs else "Sin observaciones."
        }

        # --- Calibración Óptica ---
        with st.expander(t["header_calib"], expanded=True):
            def_h = (20, 45) if tipo_logic == "pressure" else (85, 110)
            h_min, h_max = st.slider("Matiz (Hue)", 0, 179, def_h)
            s_val = st.slider("Saturación", 0, 255, (100, 255))
            v_val = st.slider("Brillo (Value)", 0, 255, (100, 255))

    # ========================
    # COLUMNA DERECHA: Análisis
    # ========================
    with col_right:
        # Tabs: análisis en vivo / historial
        tab_live, tab_hist = st.tabs([t["tab_analysis"], t["tab_history"]])

        with tab_live:
            st.subheader(t["col_tools"])

            # --- Entrada dual: cámara O archivo (nuevo) ---
            input_mode = st.radio("Fuente de imagen", ["📷 Cámara", "📁 Archivo"], horizontal=True)
            input_img = None
            if input_mode == "📷 Cámara":
                input_img = st.camera_input(t["camera_msg"])
            else:
                uploaded = st.file_uploader(t["upload_msg"], type=["png", "jpg", "jpeg", "bmp"])
                if uploaded:
                    input_img = uploaded

            # --- Procesamiento de imagen ---
            bytes_data = None
            signal = []

            if input_img:
                bytes_data = input_img.getvalue()
                img_cv = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

                if img_cv is None:
                    st.error("❌ No se pudo decodificar la imagen. Intentá con otro archivo.")
                else:
                    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
                    mask = cv2.inRange(
                        hsv,
                        np.array([h_min, s_val[0], v_val[0]]),
                        np.array([h_max, s_val[1], v_val[1]])
                    )

                    # --- Extracción de Curva con filtro de continuidad ---
                    h_px, w_px = mask.shape
                    last_valid = h_px // 2  # Inicializar en centro, no en 0 (fix bug #4)
                    for col in range(int(w_px * 0.1), int(w_px * 0.9)):
                        col_data = mask[:, col]
                        if np.max(col_data) > 0:
                            val = h_px - np.argmax(col_data)
                            if len(signal) > 0 and abs(val - last_valid) > (h_px * 0.3):
                                signal.append(last_valid)
                            else:
                                signal.append(val)
                                last_valid = val
                        else:
                            signal.append(last_valid)

                    # --- Visualización estilo monitor ---
                    if np.max(signal) > np.min(signal):
                        sig_norm = (np.array(signal) - np.min(signal)) / (np.max(signal) - np.min(signal) + 1e-6)

                        # Savitzky-Golay adaptativo (fix bug crítico)
                        win = min(31, len(sig_norm) - 1)
                        if win % 2 == 0:
                            win -= 1
                        sig_smooth = savgol_filter(sig_norm, win, 3) if win >= 5 else sig_norm

                        fig, ax = plt.subplots(figsize=(10, 3), facecolor='#0e1117')
                        ax.set_facecolor('#0e1117')
                        color_wave = "#fbbf24" if tipo_logic == "pressure" else "#22d3ee"
                        ax.plot(sig_smooth, color=color_wave, lw=2.5)
                        ax.fill_between(range(len(sig_smooth)), sig_smooth, color=color_wave, alpha=0.15)
                        ax.grid(color='#374151', linestyle='--', linewidth=0.5, alpha=0.5)
                        ax.axis('off')
                        st.pyplot(fig)
                        plt.close(fig)

                        # --- MOTOR MATEMÁTICO (implementado) ---
                        hallazgos = analizar_senal_matematica(signal, tipo_logic)

                        # Métricas de estado
                        m1, m2, m3 = st.columns(3)
                        with m1:
                            st.metric("Picos detectados", hallazgos["picos_detectados"])
                        with m2:
                            fr_est = hallazgos["frecuencia_estimada"]
                            st.metric("FR estimada", f"{fr_est} rpm" if fr_est > 0 else "N/A")
                        with m3:
                            alertas = sum([hallazgos["doble_disparo"], hallazgos["hambre_flujo"]])
                            st.metric("Alertas matemáticas", alertas,
                                      delta="⚠️ Revisar" if alertas > 0 else "✅ Sin alertas",
                                      delta_color="inverse" if alertas > 0 else "normal")

                        # Alertas visuales del motor matemático
                        if hallazgos["doble_disparo"]:
                            st.error("🔴 Motor Matemático: **Sospecha de DOBLE DISPARO** detectada.")
                        if hallazgos["hambre_flujo"]:
                            st.warning("🟠 Motor Matemático: **Posible HAMBRE DE FLUJO** detectada.")
                        if not hallazgos["doble_disparo"] and not hallazgos["hambre_flujo"]:
                            st.success("🟢 Motor Matemático: Sin asincronías obvias detectadas.")

                    else:
                        hallazgos = {"resumen": "Señal no detectada."}
                        st.warning("⚠️ No se detecta señal. Ajustá la Calibración Óptica en el panel izquierdo.")

                    st.divider()

                    # --- Botón de IA ---
                    if st.button(t["btn_ai"], use_container_width=True):
                        if not api_key:
                            st.error("⚠️ Por favor, ingresá una API Key válida en el panel izquierdo.")
                        elif not signal or np.max(signal) == np.min(signal):
                            st.error("⚠️ No hay señal válida para analizar. Ajustá la calibración primero.")
                        else:
                            with st.status("🧠 Iniciando consultoría diagnóstica...", expanded=True) as status:
                                st.write("Preparando imagen y contexto clínico...")
                                st.write(f"Pre-diagnóstico: {hallazgos.get('resumen', 'N/A')}")
                                reporte, modelo_usado = get_ai_response(
                                    bytes_data,
                                    canal,
                                    api_key,
                                    contexto_clinico,
                                    hallazgos.get("resumen", "No disponible")
                                )
                                status.update(label=f"✅ Análisis completado con {modelo_usado}", state="complete")

                            st.subheader(t["ai_report_title"])
                            if reporte.startswith("❌"):
                                st.error(reporte)
                            else:
                                st.markdown(reporte)

                                # --- Guardar en historial de sesión ---
                                entrada = {
                                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                                    "canal": canal,
                                    "modo_vent": contexto_clinico["modo_vent"],
                                    "hallazgos_matematicos": hallazgos.get("resumen", "N/A"),
                                    "reporte_ia": reporte,
                                    "modelo": modelo_usado
                                }
                                st.session_state.historial.append(entrada)
                                st.caption(f"_Análisis guardado en historial | Modelo: {modelo_usado}_")

        # --- Tab Historial ---
        with tab_hist:
            st.subheader("📋 Historial de Análisis — Sesión Actual")
            if not st.session_state.historial:
                st.info("No hay análisis registrados en esta sesión. Ejecutá una Auditoría Clínica IA para que aparezca aquí.")
            else:
                # Botón para limpiar historial
                if st.button("🗑️ Limpiar historial"):
                    st.session_state.historial = []
                    st.rerun()

                for i, entrada in enumerate(reversed(st.session_state.historial)):
                    with st.expander(
                        f"[{entrada['timestamp']}] {entrada['canal']} | {entrada['modo_vent']}",
                        expanded=(i == 0)
                    ):
                        st.caption(f"**Motor matemático:** {entrada['hallazgos_matematicos']}")
                        st.caption(f"**Modelo IA usado:** {entrada['modelo']}")
                        st.markdown(entrada["reporte_ia"])


if __name__ == "__main__":
    main()
