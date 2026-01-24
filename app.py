# Copyright (c) 2025 Ventilator Lab AI
#
# Este software es Propiedad Intelectual Privada y Confidencial.
# Desarrollado para fines educativos y de soporte a la decisión clínica.
# No constituye un dispositivo médico certificado.
#
# Todos los derechos reservados.

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

# Configuración de Matplotlib para servidores sin pantalla
matplotlib.use('Agg')

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Ventilator Lab AI",
    page_icon="🫁",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ==========================================
# 0. DICCIONARIO DE IDIOMAS & PROMPTS BLINDADOS
# ==========================================
TEXTOS = {
    "es": {
        "title": "🫁 Ventilator Lab: Híbrido",
        "subtitle": "Diagnóstico de Asincronías: **Visión Artificial + IA Generativa**",
        "sidebar_settings": "⚙️ Configuración",
        "lang_select": "Idioma / Language",
        "api_key_label": "🔑 Google API Key",
        "api_key_help": "Introduce tu clave si no tienes licencia configurada.",
        "api_warning": "Se requiere API Key para la IA.",
        "license_ok": "✅ Licencia Pro Activada",
        "api_missing": "⚠️ Falta la API Key. Configúrala en los 'Secrets' o en la barra lateral.",
        "color_calib": "🎨 Calibración de Color",
        "color_info": "Ajusta si la curva no se detecta.",
        "curve_type": "¿Qué curva vas a analizar?",
        "help_title": "❓ ¿Cómo saber cuál elegir?",
        "help_p_name": "Gráfica de PRESIÓN (Paw)",
        "help_p_desc": "🟢 Sube y baja, siempre sobre la línea base.",
        "help_f_name": "Gráfica de FLUJO (Flow)",
        "help_f_desc": "🔵 Cruza la línea del cero (Inspiración/Espiración).",
        "opt_pressure": "Presión (Paw)",
        "opt_flow": "Flujo (Flow)",
        "sliders_hue": "Matiz (H)",
        "sliders_sat": "Saturación (S)",
        "sliders_val": "Brillo (V)",
        "camera_label": "📸 Toma una foto a la pantalla",
        "debug_view": "👁️ Ver Máscara de Visión (Debug)",
        "warn_no_curve": "⚠️ El algoritmo matemático no detecta la curva. Calibra o usa la IA.",
        "math_diag_label": "📍 Análisis Geométrico:",
        "ai_section_title": "🤖 Opinión del Experto AI",
        "ai_section_desc": "Consulta a la IA para un análisis fisiológico detallado.",
        "btn_analyze": "🔍 Analizar con IA",
        "ai_success": "Reporte generado exitosamente:",
        "ai_error_auth": "❌ Error de Permisos o Modelo Vision no disponible.",
        "ai_error_conn": "❌ Error de Conexión.",
        "math_normal": "Patrón Estable (Geométrico)",
        "math_normal_desc": "No se detectaron deformaciones obvias.",
        "math_advice": "Correlacione con la clínica.",
        "diag_flow_starvation": "Posible Hambre de Flujo",
        "desc_flow_starvation": "Muesca detectada (Ratio alto).",
        "adv_flow_starvation": "Evalúe aumentar flujo o reducir Rise Time.",
        "diag_double_trigger": "Posible Doble Disparo",
        "desc_double_trigger": "Valle profundo entre ciclos rápidos.",
        "adv_double_trigger": "Evalúe Ti neural vs Ti mecánico.",
        "diag_auto_cycle": "Posible Doble Disparo/Autociclado",
        "loading_ai": "🤖 El Experto está analizando la fisiología...",
        "prompt_system": """
        Actúa como un Auditor Clínico Senior especialista en Ventilación Mecánica y Análisis Gráfico.
        Tu objetivo es identificar asincronías complejas con precisión quirúrgica.
        Responde SIEMPRE en ESPAÑOL.
        """,
        "prompt_instructions": """
        Analiza la imagen siguiendo este protocolo de razonamiento:
        
        1. ANÁLISIS DE SEÑAL: Identifica si la curva es de **{tipo_curva}**. Describe brevemente la morfología (ej. ondas cuadradas, rampa ascendente).
        2. BÚSQUEDA DE MARCADORES:
           - DOBLE DISPARO: Dos ciclos seguidos con exhalación incompleta.
           - HAMBRE DE FLUJO (Solo en PRESIÓN): Busca concavidad ("scooping") en la rama inspiratoria.
           - CICLADO PREMATURO (Solo en FLUJO): Pico positivo al inicio de la espiración.
           - ESFUERZOS INEFECTIVOS: Deflexiones positivas en fase espiratoria que no disparan ciclo.
        3. EXCLUSIÓN: Si hay mucho ruido o secreciones (ondas en serrucho), indícalo.
        
        FORMATO:
        ### 🏥 Diagnóstico Clínico: [Nombre o Trazo Normal]
        **🧐 Evidencia Morfológica:** [Descripción técnica de lo detectado]
        **🩺 Sugerencia Terapéutica:** [Recomendación clínica breve]
        """
    },
    "en": {
        "title": "🫁 Ventilator Lab: Hybrid",
        "subtitle": "Asynchrony Detection: **Computer Vision + AI**",
        "sidebar_settings": "⚙️ Settings",
        "lang_select": "Language",
        "api_key_label": "🔑 Google API Key",
        "api_key_help": "Enter your API Key.",
        "api_warning": "API Key required.",
        "license_ok": "✅ Pro License Active",
        "api_missing": "⚠️ Missing API Key.",
        "color_calib": "🎨 Color Calibration",
        "color_info": "Adjust sliders to isolate the curve.",
        "curve_type": "Curve Type",
        "help_title": "❓ How to choose?",
        "help_p_name": "PRESSURE Graph",
        "help_p_desc": "🟢 Always stays above baseline.",
        "help_f_name": "FLOW Graph",
        "help_f_desc": "🔵 Crosses zero line (In/Out).",
        "opt_pressure": "Pressure (Paw)",
        "opt_flow": "Flow",
        "sliders_hue": "Hue (H)",
        "sliders_sat": "Saturation (S)",
        "sliders_val": "Value (V)",
        "camera_label": "📸 Snapshot ventilator screen",
        "debug_view": "👁️ Debug Vision",
        "warn_no_curve": "⚠️ Curve not detected mathematically.",
        "math_diag_label": "📍 Geometric Analysis:",
        "ai_section_title": "🤖 AI Expert Opinion",
        "ai_section_desc": "Detailed physiological analysis.",
        "btn_analyze": "🔍 Analyze with AI",
        "ai_success": "Report generated:",
        "ai_error_auth": "❌ Auth Error.",
        "ai_error_conn": "❌ Connection Error.",
        "math_normal": "Stable Pattern",
        "math_normal_desc": "No major geometric issues.",
        "math_advice": "Correlate with patient status.",
        "diag_flow_starvation": "Possible Flow Starvation",
        "desc_flow_starvation": "Concavity detected.",
        "adv_flow_starvation": "Check flow settings.",
        "diag_double_trigger": "Possible Double Trigger",
        "desc_double_trigger": "Incomplete exhalation between cycles.",
        "adv_double_trigger": "Evaluate patient comfort/sedation.",
        "diag_auto_cycle": "Auto-cycling/Double Trigger",
        "loading_ai": "🤖 Analyzing morphology...",
        "prompt_system": "Act as a Senior Clinical Auditor in Mechanical Ventilation. Respond in ENGLISH.",
        "prompt_instructions": """
        Analyze image:
        1. SIGNAL: Is it **{tipo_curva}**? Describe morphology.
        2. MARKERS: Double Trigger, Flow Starvation (Pressure only), Early Cycling (Flow only), Ineffective Efforts.
        3. EXCLUSION: Check for noise/secretions.
        
        FORMAT:
        ### 🏥 Clinical Diagnosis: [Name]
        **🧐 Morphological Evidence:** [Technical description]
        **🩺 Clinical Suggestion:** [Action]
        """
    }
}

# ==========================================
# 1. LÓGICA DE IA (OPTIMIZADA)
# ==========================================

def consultar_intensivista_ia(image_bytes, tipo_curva, api_key, lang_code):
    t = TEXTOS[lang_code]
    if not api_key: return t["api_missing"]

    genai.configure(api_key=api_key)
    
    # Pre-procesamiento: Reducir tamaño para velocidad y cuota
    image_pil = Image.open(io.BytesIO(image_bytes))
    image_pil.thumbnail((1200, 1200)) # Optimización de carga

    instrucciones = t['prompt_instructions'].format(tipo_curva=tipo_curva)
    prompt_completo = f"{t['prompt_system']}\n\nINPUT:\n{instrucciones}"

    try:
        modelos_disponibles = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        prioridad = ["models/gemini-1.5-flash", "models/gemini-1.5-pro", "models/gemini-pro-vision"]
        
        modelo_final = next((m for m in prioridad if m in modelos_disponibles), modelos_disponibles[0])
        model = genai.GenerativeModel(modelo_final)
        
        with st.spinner(t['loading_ai']):
            response = model.generate_content([prompt_completo, image_pil], generation_config={"temperature": 0.1})
            return response.text
    except Exception as e:
        return f"{t['ai_error_conn']}: {str(e)}"

# ==========================================
# 2. LÓGICA MATEMÁTICA (ANTI-RUIDO)
# ==========================================

def analizar_curva_matematica(signal, tipo_curva_key, fs=50, lang_code="es"):
    t = TEXTOS[lang_code]
    hallazgos = {"diagnostico": t["math_normal"], "color": "green", "explicacion": t["math_normal_desc"], "consejo": t["math_advice"]}
    
    picos, _ = find_peaks(signal, prominence=0.15, distance=int(0.15 * fs))
    if len(picos) < 2: return hallazgos, picos

    for i in range(len(picos) - 1):
        p1, p2 = picos[i], picos[i+1]
        distancia_t = (p2 - p1) / fs
        
        if distancia_t < 1.0:
            segmento = signal[p1:p2]
            ratio_valle = np.min(segmento) / (signal[p1] + 1e-6)
            
            if tipo_curva_key == "pressure":
                if ratio_valle > 0.65: # Muesca alta
                    hallazgos.update({"diagnostico": t["diag_flow_starvation"], "color": "orange", "explicacion": t["desc_flow_starvation"], "consejo": t["adv_flow_starvation"]})
                    return hallazgos, picos
                elif ratio_valle < 0.45: # Doble disparo
                    hallazgos.update({"diagnostico": t["diag_double_trigger"], "color": "red", "explicacion": t["desc_double_trigger"], "consejo": t["adv_double_trigger"]})
                    return hallazgos, picos
            elif tipo_curva_key == "flow" and ratio_valle < 0.3:
                hallazgos.update({"diagnostico": t["diag_auto_cycle"], "color": "red"})
                return hallazgos, picos

    return hallazgos, picos

# ==========================================
# 3. INTERFAZ PRINCIPAL
# ==========================================

def main():
    if os.path.exists("logo.png"): st.sidebar.image("logo.png")
    
    st.sidebar.header("🌐 Language")
    idioma = st.sidebar.radio("Select:", ["Español", "English"], horizontal=True)
    lang = "es" if idioma == "Español" else "en"
    t = TEXTOS[lang]

    st.title(t["title"])
    st.markdown(t["subtitle"])
    
    # API Key management
    api_key = st.secrets.get("GOOGLE_API_KEY") or st.sidebar.text_input(t["api_key_label"], type="password")
    if api_key: st.sidebar.success(t["license_ok"])
    else: st.sidebar.warning(t["api_warning"])

    st.sidebar.divider()
    st.sidebar.header(t["color_calib"])
    
    # Curva y Calibración
    st.subheader(t["curve_type"])
    opcion = st.radio(" ", [t["opt_pressure"], t["opt_flow"]], horizontal=True, label_visibility="collapsed")
    tipo_logic = "pressure" if t["opt_pressure"] in opcion else "flow"

    # Presets inteligentes de color
    def_h = (20, 45) if tipo_logic == "pressure" else (80, 105)
    h_min, h_max = st.sidebar.slider(t["sliders_hue"], 0, 179, def_h)
    s_min, s_max = st.sidebar.slider(t["sliders_sat"], 0, 255, (100, 255))
    v_min, v_max = st.sidebar.slider(t["sliders_val"], 0, 255, (100, 255))

    input_img = st.camera_input(t["camera_label"])

    if input_img:
        bytes_data = input_img.getvalue()
        img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h_px, w_px, _ = img.shape

        mask = cv2.inRange(hsv, np.array([h_min, s_min, v_min]), np.array([h_max, s_max, v_max]))
        
        with st.expander(t["debug_view"]): st.image(mask)

        # Extracción de señal con filtro de continuidad
        signal = []
        last_val = 0
        for col in range(int(w_px*0.1), int(w_px*0.9)):
            col_data = mask[:, col]
            if np.max(col_data) > 0:
                current_val = h_px - np.argmax(col_data)
                # Anti-salto: si el cambio es > 30% del alto, ignorar reflejo
                if len(signal) > 0 and abs(current_val - last_val) > (h_px * 0.3):
                    signal.append(last_val)
                else:
                    signal.append(current_val)
                    last_val = current_val
            else:
                signal.append(last_val)

        if np.max(signal) > 0:
            sig_norm = (np.array(signal) - np.min(signal)) / (np.max(signal) - np.min(signal) + 1e-6)
            try: sig_clean = savgol_filter(sig_norm, 31, 3)
            except: sig_clean = sig_norm

            res_m, pks = analizar_curva_matematica(sig_clean, tipo_logic, lang_code=lang)

            # Gráfico estilo monitor
            fig, ax = plt.subplots(figsize=(10, 2.5))
            fig.patch.set_facecolor('#0e1117')
            ax.set_facecolor('black')
            ax.plot(sig_clean, color=('yellow' if tipo_logic == "pressure" else 'cyan'), lw=2)
            ax.plot(pks, sig_clean[pks], "ro" if res_m['color'] != "green" else "wo", markersize=4)
            ax.axis('off')
            st.pyplot(fig)
            st.caption(f"{t['math_diag_label']} **{res_m['diagnostico']}**")
        else:
            st.warning(t["warn_no_curve"])

        st.divider()
        if st.button(t["btn_analyze"], type="primary", use_container_width=True):
            reporte = consultar_intensivista_ia(bytes_data, tipo_logic, api_key, lang)
            st.markdown(reporte)

if __name__ == "__main__":
    main()
