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
    page_title="Asistente Ventilación AI",
    page_icon="🫁",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ==========================================
# 1. LÓGICA DE IA (ROBUSTA + AUTODESCUBRIMIENTO)
# ==========================================

def consultar_intensivista_ia(image_bytes, tipo_curva, api_key):
    """
    Conecta con Google Gemini buscando automáticamente el mejor modelo disponible.
    """
    if not api_key:
        return "⚠️ Falta la API Key. Configúrala en los 'Secrets' de Streamlit o en la barra lateral."

    # Configuración
    genai.configure(api_key=api_key)
    image_pil = Image.open(io.BytesIO(image_bytes))

    # Prompt del Sistema (Rol Médico)
    prompt = f"""
    Actúa como un Médico Intensivista experto en Ventilación Mecánica.
    Analiza esta imagen de la pantalla de un ventilador (Curva de {tipo_curva}).
    
    Tareas:
    1. Valida si la curva es legible.
    2. Busca asincronías específicas: Doble Disparo, Hambre de Flujo, Ciclado Retrasado, Esfuerzos Inefectivos.
    3. Describe brevemente la morfología visual que justifica tu hallazgo.
    4. Da una recomendación clínica concisa y segura.
    
    Formato: Responde directamente con el diagnóstico y la recomendación.
    """

    # --- ESTRATEGIA DE CONEXIÓN ROBUSTA ---
    # Intentamos listar modelos disponibles para evitar errores 404 por nombres incorrectos
    try:
        # 1. Obtener modelos disponibles para esta API Key
        lista_modelos = list(genai.list_models())
        modelos_validos = [m.name for m in lista_modelos if 'generateContent' in m.supported_generation_methods]
        
        # 2. Definir preferencias (del más rápido/nuevo al más compatible)
        preferencias = [
            "models/gemini-1.5-flash",
            "models/gemini-1.5-flash-latest",
            "models/gemini-1.5-pro",
            "models/gemini-pro-vision",
            "models/gemini-pro"
        ]
        
        modelo_elegido = None
        
        # 3. Cruzar preferencias con disponibles
        for pref in preferencias:
            if pref in modelos_validos:
                modelo_elegido = pref
                break
        
        # Fallback de emergencia: usar el primero que haya
        if not modelo_elegido and modelos_validos:
            modelo_elegido = modelos_validos[0]
            
        if not modelo_elegido:
            return f"❌ Error de Permisos: Tu API Key es válida pero Google no le permite acceder a modelos de visión.\nModelos detectados: {modelos_validos}"

        # 4. Generar Respuesta
        model = genai.GenerativeModel(modelo_elegido)
        with st.spinner(f'🤖 Analizando con {modelo_elegido}...'):
            response = model.generate_content([prompt, image_pil])
            return response.text

    except Exception as e:
        return f"❌ Error de Conexión: {str(e)}\n\nVerifica que tu API Key sea válida y tenga crédito/cuota."

# ==========================================
# 2. LÓGICA MATEMÁTICA (OPENCV - FILTROS)
# ==========================================

def analizar_curva_matematica(signal, tipo_curva, fs=50):
    hallazgos = {
        "diagnostico": "Patrón Estable (Geométrico)",
        "color": "green",
        "explicacion": "No se detectaron deformaciones obvias en el análisis matemático.",
        "consejo": "Correlacione con la clínica del paciente."
    }
    
    # Detección de Picos
    prominencia = 0.15 
    distancia_min = int(0.15 * fs)
    picos, _ = find_peaks(signal, prominence=prominencia, distance=distancia_min)
    
    if len(picos) < 2:
        return hallazgos, picos

    # Análisis de Pares
    for i in range(len(picos) - 1):
        p1 = picos[i]
        p2 = picos[i+1]
        distancia_tiempo = (p2 - p1) / fs
        
        # Si están muy cerca (< 1 seg)
        if distancia_tiempo < 1.0:
            segmento = signal[p1:p2]
            valle_idx = np.argmin(segmento)
            altura_valle = segmento[valle_idx]
            altura_pico1 = signal[p1]
            if altura_pico1 == 0: altura_pico1 = 0.001
            ratio_valle = altura_valle / altura_pico1
            
            if tipo_curva == "Presión":
                if ratio_valle > 0.6: 
                    hallazgos["diagnostico"] = "Posible Hambre de Flujo"
                    hallazgos["color"] = "orange"
                    hallazgos["explicacion"] = "Muesca detectada (Ratio alto)."
                    hallazgos["consejo"] = "Considere aumentar flujo o reducir Rise Time."
                    return hallazgos, picos
                elif ratio_valle < 0.5:
                    hallazgos["diagnostico"] = "Posible Doble Disparo"
                    hallazgos["color"] = "red"
                    hallazgos["explicacion"] = "Valle profundo entre ciclos rápidos."
                    hallazgos["consejo"] = "Evalúe Ti neural vs Ti mecánico."
                    return hallazgos, picos
            
            elif tipo_curva == "Flujo":
                if ratio_valle < 0.3:
                    hallazgos["diagnostico"] = "Posible Doble Disparo/Autociclado"
                    hallazgos["color"] = "red"
                    return hallazgos, picos

    return hallazgos, picos

# ==========================================
# 3. INTERFAZ DE USUARIO (MAIN)
# ==========================================

def main():
    st.title("🫁 Ventilator Lab: Híbrido")
    st.markdown("Diagnóstico de Asincronías: **Visión Artificial + IA Generativa**")
    
    # --- BARRA LATERAL (CONFIGURACIÓN) ---
    st.sidebar.header("⚙️ Configuración")
    
    # 1. GESTIÓN DE API KEY (SECRETS vs MANUAL)
    api_key = None
    
    # Intentamos leer el 'Secret' del servidor (Producción)
    if "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
        st.sidebar.success("✅ Licencia Pro Activada")
    else:
        # Si no hay secret, pedimos manual (Desarrollo/Usuario externo)
        api_key = st.sidebar.text_input("🔑 Google API Key", type="password", help="Introduce tu clave si no tienes licencia configurada.")
        if not api_key:
            st.sidebar.warning("Se requiere API Key para la IA.")

    st.sidebar.divider()
    
    # 2. CALIBRACIÓN DE COLOR
    st.sidebar.header("🎨 Calibración de Color")
    st.sidebar.info("Ajusta si la curva no se detecta.")
    
    tipo = st.radio("¿Qué curva estás analizando?", ["Presión (Paw)", "Flujo (Flow)"], horizontal=True)

    if "Presión" in tipo:
        def_h, def_s, def_v = (20, 40), (100, 255), (100, 255) # Amarillo
    else:
        def_h, def_s, def_v = (80, 100), (100, 255), (100, 255) # Azul

    h_min, h_max = st.sidebar.slider(f"Matiz (Hue)", 0, 179, def_h)
    s_min, s_max = st.sidebar.slider(f"Saturación (Sat)", 0, 255, def_s)
    v_min, v_max = st.sidebar.slider(f"Brillo (Val)", 0, 255, def_v)

    # --- CÁMARA Y PROCESAMIENTO ---
    imagen = st.camera_input("📸 Toma una foto a la pantalla")

    if imagen:
        bytes_data = imagen.getvalue()
        
        # A. PROCESAMIENTO VISUAL (OPENCV)
        img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, w, _ = img.shape

        # Máscara de color
        lower_color = np.array([h_min, s_min, v_min])
        upper_color = np.array([h_max, s_max, v_max])
        mask = cv2.inRange(hsv, lower_color, upper_color)

        with st.expander("👁️ Ver lo que ve la máquina (Debug)", expanded=False):
            st.image(mask, caption="Máscara Binaria", use_column_width=True)

        # Extracción de señal
        raw_signal = []
        for col in range(int(w*0.1), int(w*0.9)):
            col_data = mask[:, col]
            if np.max(col_data) > 0:
                y_pos = h - np.argmax(col_data)
                raw_signal.append(y_pos)
            else:
                val = raw_signal[-1] if len(raw_signal) > 0 else 0
                raw_signal.append(val)
        
        # Validación de señal
        signal_valid = True
        if np.max(raw_signal) == 0:
            st.warning("⚠️ El algoritmo matemático no ve la curva. Intenta calibrar los colores o usa directamente la IA.")
            signal_valid = False

        # Si hay señal, hacemos análisis matemático
        if signal_valid:
            sig_np = np.array(raw_signal)
            sig_norm = (sig_np - np.min(sig_np)) / (np.max(sig_np) - np.min(sig_np) + 1e-6)
            try:
                sig_smooth = savgol_filter(sig_norm, 31, 3)
            except:
                sig_smooth = sig_norm

            res_math, picos = analizar_curva_matematica(sig_smooth, tipo.split()[0])

            # Graficar
            fig, ax = plt.subplots(figsize=(10, 3))
            fig.patch.set_facecolor('#0e1117')
            ax.set_facecolor('black')
            color_linea = 'yellow' if "Presión" in tipo else 'cyan'
            ax.plot(sig_smooth, color=color_linea, lw=2)
            ax.plot(picos, sig_smooth[picos], "wo", markersize=5)
            ax.axis('off')
            st.pyplot(fig)
            
            st.caption(f"📍 Análisis Geométrico: **{res_math['diagnostico']}**")

        # B. ANÁLISIS POR IA GENERATIVA
        st.divider()
        col_btn, col_txt = st.columns([1, 2])
        
        with col_txt:
            st.markdown("### 🤖 Opinión del Experto")
            st.write("Consulta a la IA para un análisis clínico detallado.")
            
        with col_btn:
            consultar = st.button("🔍 Analizar con IA", type="primary")

        if consultar:
            diagnostico_ia = consultar_intensivista_ia(bytes_data, tipo, api_key)
            if "❌" in diagnostico_ia:
                st.error(diagnostico_ia)
            else:
                st.success("Reporte generado exitosamente:")
                st.info(diagnostico_ia)

if __name__ == "__main__":
    main()
