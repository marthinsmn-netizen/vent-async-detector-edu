import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
import matplotlib
matplotlib.use('Agg')

# --- Configuración de la App ---
st.set_page_config(
    page_title="Ventilador Lab - Asistente Educativo",
    page_icon="🫁",
    layout="centered"
)

# ==========================================
# LÓGICA CLÍNICA (EL CEREBRO)
# ==========================================

def analizar_curva_presion(signal, fs=50):
    """
    Analiza la forma de la curva de PRESIÓN para distinguir
    entre Hambre de Flujo (mordida) y Doble Disparo.
    """
    hallazgos = {
        "tipo": "Normal",
        "confianza": 0.0,
        "mensaje": "Patrón ventilatorio aceptable.",
        "accion": "Continuar monitorización."
    }
    
    # Detectar picos principales
    picos, _ = find_peaks(signal, prominence=0.2, distance=int(0.5 * fs))
    
    if len(picos) < 2:
        hallazgos["mensaje"] = "No se detectan ciclos suficientes."
        return hallazgos, picos

    # Evaluar doble disparo (<0.8 s entre picos)
    for i in range(len(picos) - 1):
        dt = (picos[i+1] - picos[i]) / fs
        if dt < 0.8:
            hallazgos["tipo"] = "Doble Disparo"
            hallazgos["mensaje"] = f"Dos ciclos muy cercanos: {dt:.2f} s."
            hallazgos["accion"] = "El Ti mecánico es menor que el Ti neural."
            return hallazgos, picos

    # Evaluar hambre de flujo (concavidad de la inspiración)
    idx_peak = picos[0]
    inicio = max(0, idx_peak - int(0.4 * fs))
    fin = idx_peak

    segmento = signal[inicio:fin]

    if len(segmento) > 5:
        linea_ideal = np.linspace(segmento[0], segmento[-1], len(segmento))
        diferencia = linea_ideal - segmento
        max_dep = np.max(diferencia)
        altura = np.max(signal) - np.min(signal)

        if altura > 0 and (max_dep / altura) > 0.15:
            hallazgos["tipo"] = "Hambre de Flujo"
            hallazgos["mensaje"] = "La presión se hunde durante la inspiración."
            hallazgos["accion"] = "Aumentar flujo o ajustar Rise Time."
            return hallazgos, picos

    return hallazgos, picos

# ==========================================
# INTERFAZ
# ==========================================

def main():
    st.title("🫁 Asistente de Asincronías")
    st.write("Toma una foto a la pantalla del ventilador para recibir orientación educativa.")

    modo = st.radio(
        "¿Qué curva estás viendo?",
        ["Presión (Paw)", "Flujo (Flow)"],
        horizontal=True
    )

    # Cámara
    img_file = st.camera_input("Capturar Pantalla del Ventilador")

    if img_file is not None:

        # --- Procesar imagen ---
        bytes_data = img_file.getvalue()
        img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # EXTRAER SEÑAL DIGITALIZADA
        signal = []

        for col in range(int(w * 0.1), int(w * 0.9)):
            col_data = gray[:, col]
            y_val = h - np.argmax(col_data)
            signal.append(y_val)

        signal = np.array(signal, dtype=float)

        # Normalizar señal
        signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal) + 1e-6)

        # --- Diagnóstico ---
        if "Presión" in modo:
            resultado, picos = analizar_curva_presion(signal)
        else:
            picos, _ = find_peaks(signal, prominence=0.3)
            resultado = {
                "tipo": "Análisis de Flujo",
                "mensaje": "Curva procesada. Revise el retorno a cero.",
                "accion": "Evaluar espiración completa."
            }

        st.divider()

        # --- Visualización del diagnóstico ---
        if resultado["tipo"] == "Normal":
            st.success(f"✅ **Diagnóstico:** {resultado['tipo']}")
        else:
            st.error(f"⚠️ **Diagnóstico:** {resultado['tipo']}")

        st.info(f"ℹ️ **Interpretación:** {resultado['mensaje']}")

        # --- Guía educativa ---
        with st.expander("🎓 Guía Clínica Sugerida", expanded=True):
            st.write(resultado["accion"])

        # --- Gráfico ---
        fig, ax = plt.subplots(figsize=(9, 3))
        ax.plot(signal, color='yellow' if "Presión" in modo else 'cyan', lw=2)
        ax.plot(picos, signal[picos], "rx")
        ax.set_facecolor("black")
        ax.axis("off")
        st.pyplot(fig)

if __name__ == "__main__":
    main()
