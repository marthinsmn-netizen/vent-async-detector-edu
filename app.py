import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
import matplotlib
from PIL import Image
import io
import google.generativeai as genai

# Usar backend no interactivo
matplotlib.use('Agg')

# --- Configuración Estética ---
st.set_page_config(
    page_title="Asistente Ventilación AI",
    page_icon="🫁",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ==========================================
# 1. LÓGICA DE INTELIGENCIA ARTIFICIAL (GEMINI)
# ==========================================

def consultar_intensivista_ia(image_bytes, tipo_curva, api_key):
    """
    Envía la imagen a Google Gemini 1.5 Flash actuando como médico experto.
    """
    if not api_key:
        return "⚠️ Por favor, introduce tu Google API Key en la barra lateral para usar la IA."

    try:
       # Busca esta línea:
# model = genai.GenerativeModel('gemini-1.5-flash')

# Y CÁMBIALA POR ESTA (que fuerza la versión más reciente disponible):
model = genai.GenerativeModel('gemini-1.5-flash-latest')

        # Convertir bytes a imagen PIL
        image_pil = Image.open(io.BytesIO(image_bytes))

        # El Prompt (El ROL que me pediste)
        prompt = f"""
        Actúa como un Médico Intensivista experto en Ventilación Mecánica y análisis de asincronías.
        Analiza esta imagen de la pantalla de un ventilador mecánico.
        
        Contexto: El usuario indica que es una curva de {tipo_curva}.
        
        Tu tarea:
        1. Valida si la curva es legible.
        2. Busca asincronías específicas (Doble Disparo, Hambre de Flujo, Ciclado Retrasado/Prematuro, Esfuerzos Inefectivos).
        3. Si detectas una anomalía, explica la morfología visual (ej: "se observa una muesca en la rama inspiratoria").
        4. Da una recomendación clínica breve y segura.
        
        Sé conciso. Si la curva parece normal, indícalo.
        """
        
        with st.spinner('🤖 El Intensivista AI está analizando la imagen...'):
            response = model.generate_content([prompt, image_pil])
            return response.text

    except Exception as e:
        return f"❌ Error de conexión con la IA: {str(e)}"

# ==========================================
# 2. LÓGICA MATEMÁTICA (OPENCV)
# ==========================================

def analizar_curva_matematica(signal, tipo_curva, fs=50):
    """
    Análisis rápido basado en geometría de la curva.
    """
    hallazgos = {
        "diagnostico": "Patrón Estable (Análisis Geométrico)",
        "color": "green",
        "explicacion": "No se detectaron deformaciones obvias matemáticamente.",
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
                    hallazgos["explicacion"] = "Muesca detectada por algoritmo geométrico."
                    hallazgos["consejo"] = "Considere aumentar el flujo o reducir el Rise Time."
                    return hallazgos, picos
                elif ratio_valle < 0.5:
                    hallazgos["diagnostico"] = "Posible Doble Disparo"
                    hallazgos["color"] = "red"
                    hallazgos["explicacion"] = "Doble ciclo detectado por algoritmo geométrico."
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
    st.markdown("Detección de asincronías: **Visión Artificial + IA Generativa**")

    # --- BARRA LATERAL ---
    st.sidebar.header("⚙️ Configuración")
    
    # API KEY INPUT
    api_key = st.sidebar.text_input("🔑 Google Gemini API Key", type="password", help="AIzaSyBoQ6MTvu8i725FI8rHB19qRNJGLHP0NtM")
    if not api_key:
        st.sidebar.warning("Necesitas la API Key para usar la función de 'Segunda Opinión'.")
    
    st.sidebar.divider()
    st.sidebar.header("🎨 Calibración de Color")
    st.sidebar.info("Ajusta si la curva no se detecta bien.")

    tipo = st.radio("¿Qué curva estás analizando?", ["Presión (Paw)", "Flujo (Flow)"], horizontal=True)

    # Sliders de calibración
    if "Presión" in tipo:
        def_h, def_s, def_v = (20, 40), (100, 255), (100, 255) 
    else:
        def_h, def_s, def_v = (80, 100), (100, 255), (100, 255)

    h_min, h_max = st.sidebar.slider(f"Matiz (H)", 0, 179, def_h)
    s_min, s_max = st.sidebar.slider(f"Saturación (S)", 0, 255, def_s)
    v_min, v_max = st.sidebar.slider(f"Brillo (V)", 0, 255, def_v)

    # --- CÁMARA ---
    imagen = st.camera_input("Toma una foto a la pantalla")

    if imagen:
        bytes_data = imagen.getvalue()
        
        # 1. PROCESAMIENTO MATEMÁTICO (Visualización)
        img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, w, _ = img.shape

        lower_color = np.array([h_min, s_min, v_min])
        upper_color = np.array([h_max, s_max, v_max])
        mask = cv2.inRange(hsv, lower_color, upper_color)

        with st.expander("👁️ Debug: Visión por Computadora", expanded=False):
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
        
        # Si no hay señal, avisar pero permitir intentar con IA
        signal_valid = True
        if np.max(raw_signal) == 0:
            st.warning("⚠️ El algoritmo geométrico no detectó la curva clara. Puedes intentar calibrar los colores o consultar directamente a la IA.")
            signal_valid = False

        if signal_valid:
            sig_np = np.array(raw_signal)
            sig_norm = (sig_np - np.min(sig_np)) / (np.max(sig_np) - np.min(sig_np) + 1e-6)
            try:
                sig_smooth = savgol_filter(sig_norm, 31, 3)
            except:
                sig_smooth = sig_norm

            # Análisis Matemático
            res_math, picos = analizar_curva_matematica(sig_smooth, tipo.split()[0])

            # Mostrar Gráfico
            fig, ax = plt.subplots(figsize=(10, 3))
            fig.patch.set_facecolor('#0e1117')
            ax.set_facecolor('black')
            color_linea = 'yellow' if "Presión" in tipo else 'cyan'
            ax.plot(sig_smooth, color=color_linea, lw=2)
            ax.plot(picos, sig_smooth[picos], "wo", markersize=5)
            ax.axis('off')
            st.pyplot(fig)
            
            st.caption(f"📍 Diagnóstico Geométrico: {res_math['diagnostico']}")

        # 2. CONSULTA A LA IA (EL EXPERTO)
        st.divider()
        st.subheader("🤖 Opinión del Experto (IA)")
        
        col_btn, col_info = st.columns([1, 2])
        
        with col_info:
            st.markdown("Si el gráfico de arriba no es claro o quieres una interpretación clínica detallada, consulta a la IA.")

        with col_btn:
            # Botón para llamar a la API (para no gastar tokens automáticamente)
            consultar = st.button("🔍 Analizar con IA", type="primary")

        if consultar:
            if not api_key:
                st.error("🔒 Necesitas poner la API Key en la barra lateral.")
            else:
                diagnostico_ia = consultar_intensivista_ia(bytes_data, tipo, api_key)
                st.markdown("### 📝 Reporte Clínico:")
                st.info(diagnostico_ia)

if __name__ == "__main__":
    main()
