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
# 0. DICCIONARIO DE IDIOMAS & PROMPTS ROBUSTOS
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
        "help_p_desc": "🟢 **Forma:** Sube y baja, pero **siempre se mantiene por encima de la línea base** (nunca cruza a negativo). Suele ser cuadrada o triangular.",
        "help_f_name": "Gráfica de FLUJO (Flow)",
        "help_f_desc": "🔵 **Forma:** Tiene una montaña hacia arriba (aire entrando) y una hacia abajo (aire saliendo). **Cruza la línea del cero.**",
        "opt_pressure": "Presión (Paw)",
        "opt_flow": "Flujo (Flow)",
        "sliders_hue": "Matiz (H)",
        "sliders_sat": "Saturación (S)",
        "sliders_val": "Brillo (V)",
        "camera_label": "📸 Toma una foto a la pantalla",
        "debug_view": "👁️ Ver lo que ve la máquina (Debug)",
        "warn_no_curve": "⚠️ El algoritmo matemático no ve la curva. Calibra colores o usa la IA.",
        "math_diag_label": "📍 Análisis Geométrico:",
        "ai_section_title": "🤖 Opinión del Experto",
        "ai_section_desc": "Consulta a la IA para un análisis clínico detallado.",
        "btn_analyze": "🔍 Analizar con IA",
        "ai_success": "Reporte generado exitosamente:",
        "ai_error_auth": "❌ Error de Permisos: Tu API Key es válida pero no permite visión.",
        "ai_error_conn": "❌ Error de Conexión.",
        "math_normal": "Patrón Estable (Geométrico)",
        "math_normal_desc": "No se detectaron deformaciones obvias matemáticamente.",
        "math_advice": "Correlacione con la clínica.",
        "diag_flow_starvation": "Posible Hambre de Flujo",
        "desc_flow_starvation": "Muesca detectada (Ratio alto).",
        "adv_flow_starvation": "Considere aumentar flujo o reducir Rise Time.",
        "diag_double_trigger": "Posible Doble Disparo",
        "desc_double_trigger": "Valle profundo entre ciclos rápidos.",
        "adv_double_trigger": "Evalúe Ti neural vs Ti mecánico.",
        "diag_auto_cycle": "Posible Doble Disparo/Autociclado",
        "loading_ai": "🤖 Analizando imagen (Evaluando morfología)...",
        
        # --- PROMPT CLÍNICO EN ESPAÑOL ---
        "prompt_system": """
        Actúa como un Experto Mundial en Ventilación Mecánica y Análisis de Formas de Onda (Waveform Analysis).
        Tu tarea es detectar asincronías Paciente-Ventilador con alta precisión, evitando falsos positivos.
        Responde SIEMPRE en ESPAÑOL.
        """,
        "prompt_instructions": """
        Analiza la imagen adjunta siguiendo estrictamente este protocolo de pensamiento:

        1. **Validación de Imagen:** ¿Es una curva de ventilador legible? Si es ruido o no es una pantalla, responde "Imagen no válida".
        
        2. **Identificación de Curva:** El usuario dice que es una curva de: {tipo_curva}. Verifica visualmente si coincide.
           - Presión (Paw): Generalmente positiva, forma cuadrada/rampa.
           - Flujo (Flow): Cruza la línea base (positivo insp, negativo esp).

        3. **Búsqueda de Asincronías (Criterios Estrictos):**
           - **Doble Disparo (Double Trigger):** Busca DOS ciclos inspiratorios consecutivos separados por un tiempo muy corto (<1s), donde la espiración del primero es incompleta.
           - **Hambre de Flujo (Flow Starvation):** SOLO en curvas de PRESIÓN. Busca una "muesca" o concavidad significativa en la rama inspiratoria (la presión cae cuando debería subir o mantenerse).
           - **Ciclado Retrasado (Delayed Cycling):** SOLO en curvas de PRESIÓN (Modo Soporte). Busca un pico de presión al final de la inspiración.
           - **Esfuerzos Inefectivos (Ineffective Efforts):** SOLO en curvas de FLUJO. Busca pequeñas deflexiones positivas durante la fase espiratoria que no logran disparar un nuevo ciclo.
           - **Ciclado prematuro (Early cycle):** SOLO en curvas de FLUJO. Busca pequeñas deflexiones positivas durante la fase espiratoria, muy proximas a la inspiración que no logran disparar un nuevo ciclo, y ausencia del pico negativo de flujo.

        4. **Conclusión:**
           - Si la curva se ve normal y limpia, di "Trazo Normo-funcional". No inventes problemas.
           - Si encuentras algo, describe la morfología (ej: "Se observa concavidad en el tercio medio de la inspiración").
           - Da una recomendación clínica breve (ej: "Aumentar Flow / Ajustar Rise Time").

        FORMATO DE RESPUESTA:
        **Diagnóstico:** [Nombre de la asincronía o "Normal"]
        **Hallazgo Visual:** [Descripción técnica breve de lo que ves]
        **Acción Sugerida:** [Ajuste del ventilador recomendado]
        """
    },
    "en": {
        "title": "🫁 Ventilator Lab: Hybrid",
        "subtitle": "Asynchrony Detection: **Computer Vision + Generative AI**",
        "sidebar_settings": "⚙️ Settings",
        "lang_select": "Language",
        "api_key_label": "🔑 Google API Key",
        "api_key_help": "Enter your key if no license is configured.",
        "api_warning": "API Key required for AI features.",
        "license_ok": "✅ Pro License Active",
        "api_missing": "⚠️ Missing API Key. Configure in 'Secrets' or sidebar.",
        "color_calib": "🎨 Color Calibration",
        "color_info": "Adjust if the curve is not detected.",
        "curve_type": "Which curve are you analyzing?",
        "help_title": "❓ How to identify the curve?",
        "help_p_name": "PRESSURE Graph (Paw)",
        "help_p_desc": "🟢 **Shape:** Goes up and down but **stays above the baseline** (never goes negative). Usually square or triangular.",
        "help_f_name": "FLOW Graph (Flow)",
        "help_f_desc": "🔵 **Shape:** Has a wave going Up (Inspiration) and Down (Expiration). **It crosses the zero line.**",
        "opt_pressure": "Pressure (Paw)",
        "opt_flow": "Flow",
        "sliders_hue": "Hue (H)",
        "sliders_sat": "Saturation (S)",
        "sliders_val": "Value (V)",
        "camera_label": "📸 Take a picture of the screen",
        "debug_view": "👁️ Machine Vision View (Debug)",
        "warn_no_curve": "⚠️ Math algorithm cannot see the curve. Calibrate colors or use AI.",
        "math_diag_label": "📍 Geometric Analysis:",
        "ai_section_title": "🤖 Expert Opinion",
        "ai_section_desc": "Consult AI for detailed clinical analysis.",
        "btn_analyze": "🔍 Analyze with AI",
        "ai_success": "Report generated successfully:",
        "ai_error_auth": "❌ Permission Error: Valid Key but Vision models not allowed.",
        "ai_error_conn": "❌ Connection Error.",
        "math_normal": "Stable Pattern (Geometric)",
        "math_normal_desc": "No obvious deformations detected mathematically.",
        "math_advice": "Correlate with clinical status.",
        "diag_flow_starvation": "Possible Flow Starvation",
        "desc_flow_starvation": "Concavity detected (High Ratio).",
        "adv_flow_starvation": "Consider increasing flow or reducing Rise Time.",
        "diag_double_trigger": "Possible Double Trigger",
        "desc_double_trigger": "Deep valley between fast cycles.",
        "adv_double_trigger": "Evaluate Neural Ti vs Mechanical Ti.",
        "diag_auto_cycle": "Possible Double Trigger/Auto-cycling",
        "loading_ai": "🤖 Analyzing image (Evaluating morphology)...",
        
        # --- PROMPT CLÍNICO EN INGLÉS ---
        "prompt_system": """
        Act as a World-Class Expert in Mechanical Ventilation and Waveform Analysis.
        Your task is to detect Patient-Ventilator asynchronies with high precision, avoiding false positives.
        Respond ALWAYS in ENGLISH.
        """,
        "prompt_instructions": """
        Analyze the attached image following this strict thinking protocol:

        1. **Image Validation:** Is this a legible ventilator waveform? If it's noise or not a screen, reply "Invalid Image".
        
        2. **Curve Identification:** The user states this is a: {tipo_curva} curve. Visually verify if this matches.
           - Pressure (Paw): Generally positive, square/ramp shape.
           - Flow: Crosses baseline (positive insp, negative exp).

        3. **Asynchrony Search (Strict Criteria):**
           - **Double Trigger:** Look for TWO consecutive inspiratory cycles separated by a very short time (<1s), with incomplete exhalation of the first.
           - **Flow Starvation:** ONLY in PRESSURE curves. Look for a "scooping" or significant concavity in the inspiratory limb (pressure drops when it should rise or plateau).
           - **Delayed Cycling:** ONLY in PRESSURE curves (Support Mode). Look for a pressure spike at the very end of inspiration.
           - **Ineffective Efforts:** ONLY in FLOW curves. Look for small positive deflections during the expiratory phase that fail to trigger a new cycle.
           - **Early Cycling:** ONLY in FLOW curves. Look for small positive deflections during the expiratory phase, very close to inspiration, that fail to trigger a new cycle, and absence of the negative flow peak.

        4. **Conclusion:**
           - If the curve looks normal and clean, say "Normal Functional Trace". Do not invent problems.
           - If you find something, describe the morphology (e.g., "Concavity observed in mid-inspiration").
           - Provide a brief clinical recommendation (e.g., "Increase Flow / Adjust Rise Time").

        RESPONSE FORMAT:
        **Diagnosis:** [Asynchrony Name or "Normal"]
        **Visual Finding:** [Brief technical description]
        **Suggested Action:** [Ventilator adjustment]
        """
    }
}

# ==========================================
# 1. LÓGICA DE IA (ROBUSTA + MULTILINGÜE)
# ==========================================

def consultar_intensivista_ia(image_bytes, tipo_curva, api_key, lang_code):
    t = TEXTOS[lang_code]
    
    if not api_key:
        return t["api_missing"]

    genai.configure(api_key=api_key)
    image_pil = Image.open(io.BytesIO(image_bytes))

    # Construcción del Prompt con Variables
    instrucciones = t['prompt_instructions'].format(tipo_curva=tipo_curva)
    
    prompt_completo = f"""
    {t['prompt_system']}
    
    INPUT DATA:
    {instrucciones}
    """

    try:
        lista_modelos = list(genai.list_models())
        modelos_validos = [m.name for m in lista_modelos if 'generateContent' in m.supported_generation_methods]
        
        preferencias = [
            "models/gemini-1.5-flash",
            "models/gemini-1.5-flash-latest",
            "models/gemini-1.5-pro",
            "models/gemini-pro-vision",
            "models/gemini-pro"
        ]
        
        modelo_elegido = None
        for pref in preferencias:
            if pref in modelos_validos:
                modelo_elegido = pref
                break
        
        if not modelo_elegido and modelos_validos:
            modelo_elegido = modelos_validos[0]
            
        if not modelo_elegido:
            return t["ai_error_auth"]

        model = genai.GenerativeModel(modelo_elegido)
        with st.spinner(t['loading_ai']):
            response = model.generate_content([prompt_completo, image_pil])
            return response.text

    except Exception as e:
        return f"{t['ai_error_conn']}: {str(e)}"

# ==========================================
# 2. LÓGICA MATEMÁTICA
# ==========================================

def analizar_curva_matematica(signal, tipo_curva_key, fs=50, lang_code="es"):
    t = TEXTOS[lang_code]
    
    hallazgos = {
        "diagnostico": t["math_normal"],
        "color": "green",
        "explicacion": t["math_normal_desc"],
        "consejo": t["math_advice"]
    }
    
    prominencia = 0.15 
    distancia_min = int(0.15 * fs)
    picos, _ = find_peaks(signal, prominence=prominencia, distance=distancia_min)
    
    if len(picos) < 2:
        return hallazgos, picos

    for i in range(len(picos) - 1):
        p1 = picos[i]
        p2 = picos[i+1]
        distancia_tiempo = (p2 - p1) / fs
        
        if distancia_tiempo < 1.0:
            segmento = signal[p1:p2]
            valle_idx = np.argmin(segmento)
            altura_valle = segmento[valle_idx]
            altura_pico1 = signal[p1]
            if altura_pico1 == 0: altura_pico1 = 0.001
            ratio_valle = altura_valle / altura_pico1
            
            if tipo_curva_key == "pressure":
                if ratio_valle > 0.6: 
                    hallazgos["diagnostico"] = t["diag_flow_starvation"]
                    hallazgos["color"] = "orange"
                    hallazgos["explicacion"] = t["desc_flow_starvation"]
                    hallazgos["consejo"] = t["adv_flow_starvation"]
                    return hallazgos, picos
                elif ratio_valle < 0.5:
                    hallazgos["diagnostico"] = t["diag_double_trigger"]
                    hallazgos["color"] = "red"
                    hallazgos["explicacion"] = t["desc_double_trigger"]
                    hallazgos["consejo"] = t["adv_double_trigger"]
                    return hallazgos, picos
            
            elif tipo_curva_key == "flow":
                if ratio_valle < 0.3:
                    hallazgos["diagnostico"] = t["diag_auto_cycle"]
                    hallazgos["color"] = "red"
                    return hallazgos, picos

    return hallazgos, picos

# ==========================================
# 3. INTERFAZ DE USUARIO (MAIN)
# ==========================================

def main():
    # --- BARRA LATERAL (CONFIGURACIÓN) ---
    st.sidebar.header("🌐 Language / Idioma")
    idioma_selec = st.sidebar.radio("Select:", ["Español", "English"], horizontal=True)
    
    lang = "es" if idioma_selec == "Español" else "en"
    t = TEXTOS[lang]

    st.title(t["title"])
    st.markdown(t["subtitle"])
    
    st.sidebar.divider()
    st.sidebar.header(t["sidebar_settings"])
    
    # 1. GESTIÓN DE API KEY
    api_key = None
    if "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
        st.sidebar.success(t["license_ok"])
    else:
        api_key = st.sidebar.text_input(t["api_key_label"], type="password", help=t["api_key_help"])
        if not api_key:
            st.sidebar.warning(t["api_warning"])

    st.sidebar.divider()
    
    # 2. CALIBRACIÓN
    st.sidebar.header(t["color_calib"])
    st.sidebar.info(t["color_info"])
    
    # 3. SELECCIÓN DE CURVA
    st.subheader(t["curve_type"])

    with st.expander(t["help_title"]):
        st.info(f"{t['help_p_name']}\n\n{t['help_p_desc']}")
        st.info(f"{t['help_f_name']}\n\n{t['help_f_desc']}")

    opcion_curva = st.radio(" ", [t["opt_pressure"], t["opt_flow"]], horizontal=True, label_visibility="collapsed")
    
    tipo_logica = "pressure" if t["opt_pressure"] in opcion_curva else "flow"

    if tipo_logica == "pressure":
        def_h, def_s, def_v = (20, 40), (100, 255), (100, 255) 
    else:
        def_h, def_s, def_v = (80, 100), (100, 255), (100, 255)

    h_min, h_max = st.sidebar.slider(t["sliders_hue"], 0, 179, def_h)
    s_min, s_max = st.sidebar.slider(t["sliders_sat"], 0, 255, def_s)
    v_min, v_max = st.sidebar.slider(t["sliders_val"], 0, 255, def_v)

    # --- CÁMARA ---
    imagen = st.camera_input(t["camera_label"])

    if imagen:
        bytes_data = imagen.getvalue()
        
        img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, w, _ = img.shape

        lower_color = np.array([h_min, s_min, v_min])
        upper_color = np.array([h_max, s_max, v_max])
        mask = cv2.inRange(hsv, lower_color, upper_color)

        with st.expander(t["debug_view"], expanded=False):
            st.image(mask, caption="Mask", use_column_width=True)

        raw_signal = []
        for col in range(int(w*0.1), int(w*0.9)):
            col_data = mask[:, col]
            if np.max(col_data) > 0:
                y_pos = h - np.argmax(col_data)
                raw_signal.append(y_pos)
            else:
                val = raw_signal[-1] if len(raw_signal) > 0 else 0
                raw_signal.append(val)
        
        signal_valid = True
        if np.max(raw_signal) == 0:
            st.warning(t["warn_no_curve"])
            signal_valid = False

        if signal_valid:
            sig_np = np.array(raw_signal)
            sig_norm = (sig_np - np.min(sig_np)) / (np.max(sig_np) - np.min(sig_np) + 1e-6)
            try:
                sig_smooth = savgol_filter(sig_norm, 31, 3)
            except:
                sig_smooth = sig_norm

            res_math, picos = analizar_curva_matematica(sig_smooth, tipo_logica, fs=50, lang_code=lang)

            fig, ax = plt.subplots(figsize=(10, 3))
            fig.patch.set_facecolor('#0e1117')
            ax.set_facecolor('black')
            color_linea = 'yellow' if tipo_logica == "pressure" else 'cyan'
            ax.plot(sig_smooth, color=color_linea, lw=2)
            ax.plot(picos, sig_smooth[picos], "wo", markersize=5)
            ax.axis('off')
            st.pyplot(fig)
            
            st.caption(f"{t['math_diag_label']} **{res_math['diagnostico']}**")

        st.divider()
        col_btn, col_txt = st.columns([1, 2])
        
        with col_txt:
            st.markdown(f"### {t['ai_section_title']}")
            st.write(t["ai_section_desc"])
            
        with col_btn:
            consultar = st.button(t["btn_analyze"], type="primary")

        if consultar:
            diagnostico_ia = consultar_intensivista_ia(bytes_data, tipo_logica, api_key, lang)
            if "❌" in diagnostico_ia:
                st.error(diagnostico_ia)
            else:
                st.success(t["ai_success"])
                st.info(diagnostico_ia)

if __name__ == "__main__":
    main()
