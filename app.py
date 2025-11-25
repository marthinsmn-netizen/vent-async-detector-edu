import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
import matplotlib
matplotlib.use('Agg')

# ----------------------------
# CONFIG STREAMLIT
# ----------------------------
st.set_page_config(
    page_title="Ventilator Lab - AI Assistant",
    page_icon="🫁",
    layout="centered"
)

# ===============================
# CSS ESPECIAL PARA iPHONE / SAFARI
# ===============================
st.markdown("""
<style>

    /* 🔵 Fix general para iPhone (Safari boosting issue) */
    html, body, [class*="css"] {
        -webkit-text-size-adjust: 100% !important;
        text-size-adjust: 100% !important;
        color: #0A1A2F !important;
        font-weight: 600 !important;
    }

    /* 🔵 Fondo principal */
    .main {
        background-color: #F7FAFF !important;
    }

    /* 🔵 Títulos muy blancos → corregidos */
    h1, h2, h3, h4 {
        color: #0A1A2F !important;
        font-weight: 800 !important;
        text-shadow: none !important;
    }

    /* 🔵 Párrafos */
    p, span, label, div {
        color: #0A1A2F !important;
    }

    /* 🔵 Cards / contenedores */
    .stContainer, .stCard, .stAlert {
        background: #FFFFFF !important;
        border-radius: 14px !important;
        border: 1px solid #F0F0F0 !important;
        color: #0A1A2F !important;
    }

    /* 🔵 Inputs y botones */
    input, textarea {
        color: #0A1A2F !important;
        background: white !important;
        border-radius: 10px !important;
    }

    /* 🔵 Expander claro */
    .streamlit-expanderHeader {
        background-color: #E9F1FF !important;
        color: #0A1A2F !important;
        border-radius: 8px !important;
        padding: 8px !important;
    }

    /* 🔵 Versión Mobile: tamaños y contraste */
    @media screen and (max-width: 480px) {

        h1 {
            font-size: 1.7rem !important;
            color: #0A1A2F !important;
        }

        h2 {
            font-size: 1.4rem !important;
        }
        
        p, span, label, li {
            font-size: 1rem !important;
            color: #0A1A2F !important;
        }

        .stButton button {
            font-size: 1.1rem !important;
            padding: 0.6rem 1.2rem !important;
        }

        /* Evitar textos translúcidos en iPhone */
        * {
            opacity: 1 !important;
        }

        /* Forzar contraste dentro de contenedores */
        .stContainer, .stCard, .stAlert {
            background: #FFFFFF !important;
            color: #0A1A2F !important;
        }
    }

</style>
""", unsafe_allow_html=True)

# ==========================================
# LÓGICA CLÍNICA (EL CEREBRO)
# ==========================================

def analizar_curva_presion(signal, fs=50):
    hallazgos = {
        "tipo": "Normal",
        "confianza": 0.0,
        "mensaje": "Patrón ventilatorio aceptable.",
        "accion": "Continuar monitorización."
    }
    
    picos, _ = find_peaks(signal, prominence=0.2, distance=int(0.5*fs))
    
    if len(picos) < 2:
        hallazgos["mensaje"] = "No se detectan suficientes ciclos para un diagnóstico."
        return hallazgos, picos

    for i in range(len(picos)-1):
        tiempo_entre_picos = (picos[i+1] - picos[i]) / fs
        if tiempo_entre_picos < 0.8:
            hallazgos["tipo"] = "Doble Disparo"
            hallazgos["mensaje"] = f"Se detectaron dos ciclos en {tiempo_entre_picos:.2f} segundos."
            hallazgos["accion"] = "Posible tiempo inspiratorio corto. El paciente quiere más aire o tiempo."
            return hallazgos, picos

    inicio = max(0, picos - int(0.4*fs))
    fin = picos
    segmento = signal[inicio:fin]
    
    if len(segmento) > 5:
        linea_ideal = np.linspace(segmento, segmento[-1], len(segmento))
        diferencia = linea_ideal - segmento
        max_depresion = np.max(diferencia)
        altura_pico = np.max(signal) - np.min(signal)
        
        if (max_depresion / altura_pico) > 0.15:
            hallazgos["tipo"] = "Hambre de Flujo"
            hallazgos["mensaje"] = "La curva de presión se hunde durante la entrada de aire."
            hallazgos["accion"] = "El flujo es insuficiente para la demanda del paciente."
            return hallazgos, picos

    return hallazgos, picos

# ==========================================
# INTERFAZ PARA EL ALUMNO
# ==========================================

def main():
    st.title("🫁 Ventilator Lab - AI Assistant")
    st.write("Herramienta educativa para analizar asincronías paciente-ventilador.")

    modo = st.radio("¿Qué curva estás viendo?", ["Presión (Paw)", "Flujo (Flow)"], horizontal=True)

    img_file = st.camera_input("📸 Capturar Pantalla del Ventilador")

    if img_file is not None:
        bytes_data = img_file.getvalue()
        img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        h, w = gray.shape
        signal = []

        for col in range(int(w*0.1), int(w*0.9)):
            col_data = gray[:, col]
            y_val = h - np.argmax(col_data)
            signal.append(y_val)
        
        signal = np.array(signal)
        signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal) + 1e-6)

        if "Presión" in modo:
            resultado, picos = analizar_curva_presion(signal)
        else:
            picos, _ = find_peaks(signal, prominence=0.3)
            resultado = {"tipo": "Análisis de Flujo", "mensaje": "Visualizando ciclos.", "accion": "Verifique retorno a cero."}

        st.divider()

        if resultado["tipo"] == "Normal":
            st.success(f"✅ Diagnóstico: {resultado['tipo']}")
        else:
            st.error(f"⚠️ Diagnóstico: {resultado['tipo']}")
        
        st.info(f"ℹ️ {resultado['mensaje']}")

        with st.expander("🎓 Guía Clínica"):
            if resultado["tipo"] == "Doble Disparo":
                st.markdown("""
                **El paciente realiza dos esfuerzos seguidos.**
                **Acciones:**
                - Aumentar Ti
                - Aumentar volumen tidal
                - Evaluar analgesia/sedación
                """)
            elif resultado["tipo"] == "Hambre de Flujo":
                st.markdown("""
                **El paciente demanda más flujo.**
                **Acciones:**
                - Aumentar flujo inspiratorio
                - Acelerar rise time
                - Considerar PS
                """)
            else:
                st.markdown("**Patrón estable. Continuar monitorización.**")

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(signal, color='yellow' if "Presión" in modo else 'cyan', lw=2)
        ax.plot(picos, signal[picos], "x", color='red')
        ax.set_facecolor('#000000')
        fig.patch.set_facecolor('#0e1117')
        ax.axis('off')
        st.pyplot(fig)

if __name__ == "__main__":
    main()
