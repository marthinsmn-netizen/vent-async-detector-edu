# ==========================================
# BLOQUE 1: Importaciones y Configuración
# ==========================================
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter

# Configuración crítica para entornos sin display
import matplotlib
matplotlib.use('Agg')

# Configuración de página
st.set_page_config(
    page_title="Detector de Asincronías Ventilatorias",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# BLOQUE 2: Funciones Core de Procesamiento
# ==========================================
def analyze_double_trigger(signal_data, sample_rate=50, sensitivity=0.5):
    """
    Detecta eventos de Doble Disparo en una señal unidimensional de ventilación.
    """

    # Diccionario de resultados correctamente indentado y con listas vacías
    results = {
        "detected": False,
        "event_count": 0,
        "events": [],
        "peaks": [],
        "signal_processed": None,
        "message": ""
    }

    # --- Paso 1: Preprocesamiento y suavizado ---
    try:
        window = 11
        poly = 3
        smoothed = savgol_filter(signal_data, window_length=window, polyorder=poly)
    except Exception:
        smoothed = signal_data

    results["signal_processed"] = smoothed

    # --- Paso 2: Normalización ---
    sig_min, sig_max = np.min(smoothed), np.max(smoothed)
    if sig_max - sig_min == 0:
        results["message"] = "Señal plana o sin variación detectada."
        return results

    norm_sig = (smoothed - sig_min) / (sig_max - sig_min)

    # --- Paso 3: Configuración de parámetros ---
    prominence_val = max(0.1, 0.6 - (sensitivity * 0.5))
    min_dist_samples = int(0.2 * sample_rate)
    min_width_samples = int(0.05 * sample_rate)

    peaks, properties = find_peaks(
        norm_sig,
        prominence=prominence_val,
        distance=min_dist_samples,
        width=min_width_samples
    )
    results["peaks"] = peaks

    # --- Paso 4: Detección de doble disparo ---
    dt_threshold_seconds = 1.0
    dt_threshold_samples = dt_threshold_seconds * sample_rate

    dt_events = []  # lista vacía corregida

    if len(peaks) >= 2:
        for i in range(len(peaks) - 1):
            idx_current = peaks[i]
            idx_next = peaks[i + 1]

            interval_samples = idx_next - idx_current
            interval_seconds = interval_samples / sample_rate

            # Criterio 1: proximidad temporal
            if interval_samples < dt_threshold_samples:

                # Criterio 2: profundidad del valle
                segment = norm_sig[idx_current:idx_next]
                if len(segment) > 0:
                    valley_min = np.min(segment)

                    event_data = {
                        "peak1": idx_current,
                        "peak2": idx_next,
                        "interval_sec": interval_seconds,
                        "stacking_idx": float(valley_min)
                    }
                    dt_events.append(event_data)

    results["events"] = dt_events
    results["event_count"] = len(dt_events)
    results["detected"] = len(dt_events) > 0

    return results
    def analyze_ineffective_efforts(signal_data, major_peaks, sample_rate=50):
    """
    Detecta esfuerzos inefectivos buscando pequeñas perturbaciones positivas
    durante la fase espiratoria (entre dos picos principales).
    """
    ie_events =
    
    # Si tenemos menos de 2 respiraciones, no podemos analizar intervalos
    if len(major_peaks) < 2:
        return ie_events

    # Recorremos el espacio ENTRE cada par de picos principales
    for i in range(len(major_peaks) - 1):
        start_idx = major_peaks[i]
        end_idx = major_peaks[i+1]
        
        # Definimos la "Zona de Exhalación" (aprox. el 50-90% del intervalo)
        # para evitar confundirnos con el inicio/final de la respiración principal
        interval_len = end_idx - start_idx
        search_start = start_idx + int(interval_len * 0.2) # Saltamos el descenso inicial
        search_end = end_idx - int(interval_len * 0.1)     # Saltamos el ascenso final
        
        segment = signal_data[search_start:search_end]
        
        if len(segment) == 0: continue

        # Buscamos "micro-picos" en este segmento de exhalación
        # Usamos una prominencia mucho menor porque estos esfuerzos son débiles
        micro_peaks, _ = find_peaks(segment, prominence=0.05, width=5)
        
        if len(micro_peaks) > 0:
            # Si encontramos algo, ajustamos el índice para que coincida con la señal original
            absolute_idx = search_start + micro_peaks
            ie_events.append(absolute_idx)
            
    return ie_events

# ==========================================
# BLOQUE 3: Interfaz de Usuario
# ==========================================

def main():
    st.title("🩺 Detección de Asincronías: Fase 2")
    st.markdown("""
    Este módulo analiza formas de onda capturadas del ventilador para detectar **Doble Disparo**.
    Asegúrese de capturar una imagen clara donde la curva sea visible.
    """)

    # Sidebar
    st.sidebar.header("Configuración del Algoritmo")
    sensibilidad = st.sidebar.slider("Sensibilidad de Detección", 0.0, 1.0, 0.5)
    fs_estimada = st.sidebar.number_input("Frecuencia de Muestreo Estimada (px/s)", min_value=10, value=50, step=10)

    # Captura de cámara
    img_buffer = st.camera_input("Capturar Pantalla del Ventilador")

    if img_buffer is not None:
        bytes_data = img_buffer.getvalue()
        img_array = np.frombuffer(bytes_data, np.uint8)
        original_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if original_img is not None:
            st.image(original_img, caption="Imagen Capturada", channels="BGR", use_column_width=True)

            with st.spinner("Procesando imagen y extrayendo señal..."):
                gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
                signal_extracted = []

                height, width = gray.shape
                start_col = int(width * 0.1)
                end_col = int(width * 0.9)

                for col in range(start_col, end_col):
                    column_data = gray[:, col]
                    max_idx = np.argmax(column_data)
                    y_val = height - max_idx
                    signal_extracted.append(y_val)

                signal_np = np.array(signal_extracted)

      # --- Análisis Fase 2 (Doble Disparo) ---
            analysis = analyze_double_trigger(signal_np, sample_rate=fs_estimada, sensitivity=sensibilidad)
            
            # --- Análisis Fase 3 (Esfuerzos Inefectivos) ---
            # Usamos los picos detectados en la fase 2 para buscar entre ellos
            ie_peaks = analyze_ineffective_efforts(analysis["signal_processed"], analysis["peaks"], sample_rate=fs_estimada)

            # --- Visualización de Resultados ---
            st.divider()
            st.subheader("Resultados del Análisis Multimodal")

            # Métricas (Actualizadas con Fase 3)
            m1, m2, m3 = st.columns(3)
            m1.metric("Ciclos Ventilatorios", len(analysis["peaks"]))
            
            m2.metric("Doble Disparo", analysis["event_count"], 
                        delta="-Alerta" if analysis["detected"] else "Normal",
                        delta_color="inverse")
            
            m3.metric("Esfuerzos Inefectivos", len(ie_peaks),
                        delta="-Fatiga Muscular" if len(ie_peaks) > 0 else "Normal",
                        delta_color="inverse")

            # --- Gráfico Combinado ---
            fig, ax = plt.subplots(figsize=(10, 5))
            
            # 1. Señal Base
            ax.plot(analysis["signal_processed"], label='Presión/Flujo', color='steelblue', linewidth=1.5)
            
            # 2. Picos Principales (Respiraciones)
            peaks_x = analysis["peaks"]
            peaks_y = analysis["signal_processed"][peaks_x]
            ax.scatter(peaks_x, peaks_y, color='lime', s=60, label='Disparo Ventilador', zorder=5)

            # 3. Doble Disparo (Líneas Rojas)
            if analysis["detected"]:
                for event in analysis["events"]:
                    p1, p2 = event["peak1"], event["peak2"]
                    ax.plot([p1, p2], [analysis["signal_processed"][p1], analysis["signal_processed"][p2]], 
                            color='red', linewidth=3, linestyle='--')
                    ax.text(p2, analysis["signal_processed"][p2] + 0.05, "DT", color='red', fontsize=10, fontweight='bold')

            # 4. Esfuerzos Inefectivos (Marcadores Naranjas)
            if len(ie_peaks) > 0:
                ie_y = analysis["signal_processed"][ie_peaks]
                ax.scatter(ie_peaks, ie_y, color='orange', marker='x', s=100, label='Esfuerzo Inefectivo', zorder=6)
                for idx in ie_peaks:
                     ax.text(idx, analysis["signal_processed"][idx] + 0.05, "IE", color='orange', fontsize=9)

            ax.set_title("Análisis de Asincronías: DT + IE")
            ax.set_xlabel("Tiempo (muestras)")
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

            if analysis["detected"]:
                st.warning("""
                ⚠️ **Alerta:** Se han detectado eventos compatibles con Doble Disparo. Considerar cambios. En VC-CMV evaluar reverse trigger, timepo inspiratorio insuficiente. Considerar cambiar de modo  PC-CMV
                """)
            else:
                st.success("Análisis completado: No se detectaron asincronías mayores.")

        else:
            st.error("Error: No se pudo decodificar la imagen.")
    else:
        st.info("Esperando captura de imagen...")


if __name__ == "__main__":
    main()
