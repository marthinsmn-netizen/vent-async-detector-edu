# ==========================================
# BLOQUE 1: Importaciones y Configuración
# ==========================================
import streamlit as st
import cv2
import numpy as np
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
    signal_data: 1D array-like (list o np.array)
    sample_rate: muestras por segundo (px/s)
    sensitivity: float [0..1] (mayor = detecta picos más sutiles)
    Retorna un dict con keys: detected (bool), event_count (int), events (list),
    peaks (list de índices), signal_processed (np.array), message (str)
    """
    # Normalizar input
    signal = np.asarray(signal_data, dtype=float)
    results = {
        "detected": False,
        "event_count": 0,
        "events": [],
        "peaks": [],
        "signal_processed": None,
        "message": ""
    }

    if signal.size == 0:
        results["message"] = "Señal vacía."
        return results

    # --- Paso 1: Preprocesamiento y suavizado ---
    try:
        # Ajustamos ventana para que sea impar y < longitud de señal
        window = 11
        if window >= signal.size:
            window = signal.size - 1 if (signal.size - 1) % 2 == 1 else signal.size - 2
            if window < 3:
                window = 3
        poly = 3
        smoothed = savgol_filter(signal, window_length=window, polyorder=min(poly, window-1))
    except Exception:
        smoothed = signal.copy()

    results["signal_processed"] = smoothed

    # --- Paso 2: Normalización ---
    sig_min, sig_max = np.min(smoothed), np.max(smoothed)
    if sig_max - sig_min == 0:
        results["message"] = "Señal plana o sin variación detectada."
        return results

    norm_sig = (smoothed - sig_min) / (sig_max - sig_min)

    # --- Paso 3: Configuración de parámetros para find_peaks ---
    prominence_val = max(0.01, 0.6 - (sensitivity * 0.5))  # margen mínimo menor para señales débiles
    min_dist_samples = max(1, int(0.2 * sample_rate))
    min_width_samples = max(1, int(0.05 * sample_rate))

    peaks, properties = find_peaks(
        norm_sig,
        prominence=prominence_val,
        distance=min_dist_samples,
        width=min_width_samples
    )

    peaks = np.asarray(peaks, dtype=int)
    results["peaks"] = peaks.tolist()

    # --- Paso 4: Detección de doble disparo ---
    dt_threshold_seconds = 1.0
    dt_threshold_samples = dt_threshold_seconds * sample_rate

    dt_events = []

    if peaks.size >= 2:
        for i in range(peaks.size - 1):
            idx_current = int(peaks[i])
            idx_next = int(peaks[i + 1])

            interval_samples = idx_next - idx_current
            interval_seconds = interval_samples / float(sample_rate)

            # Criterio 1: proximidad temporal
            if interval_samples < dt_threshold_samples and interval_samples > 0:
                # Criterio 2: profundidad del valle (stacking)
                # Usamos segmento exclusivo entre picos
                start_seg = idx_current
                end_seg = idx_next
                if end_seg - start_seg <= 1:
                    continue
                segment = norm_sig[start_seg:end_seg]
                if segment.size > 0:
                    valley_min = float(np.min(segment))
                    event_data = {
                        "peak1": idx_current,
                        "peak2": idx_next,
                        "interval_sec": interval_seconds,
                        "stacking_idx": valley_min
                    }
                    dt_events.append(event_data)

    results["events"] = dt_events
    results["event_count"] = len(dt_events)
    results["detected"] = len(dt_events) > 0

    return results


def analyze_ineffective_efforts(signal_data, major_peaks, sample_rate=50):
    """
    Detecta esfuerzos inefectivos (IE) buscando pequeñas perturbaciones positivas
    durante la fase espiratoria (entre dos picos principales).
    Retorna lista de índices absolutos donde se detectaron micro-picos (IE).
    """
    signal = np.asarray(signal_data, dtype=float)

    ie_events = []

    # Validación básica
    if signal.size == 0:
        return ie_events

    major_peaks_arr = np.asarray(major_peaks, dtype=int)
    if major_peaks_arr.size < 2:
        return ie_events

    for i in range(major_peaks_arr.size - 1):
        start_idx = int(major_peaks_arr[i])
        end_idx = int(major_peaks_arr[i + 1])

        # Protecciones por rango
        if end_idx <= start_idx + 2:
            continue

        interval_len = end_idx - start_idx
        search_start = start_idx + int(interval_len * 0.2)  # evitamos inicio
        search_end = end_idx - int(interval_len * 0.1)      # evitamos final

        # Aseguramos bounds
        search_start = max(0, min(search_start, signal.size - 1))
        search_end = max(search_start + 1, min(search_end, signal.size))

        segment = signal[search_start:search_end]
        if segment.size == 0:
            continue

        # Buscar micro-picos (prominencia pequeña). Ajusta parámetros según señal real.
        micro_peaks = find_peaks(segment, prominence=0.03, width=2)[0]

        if micro_peaks.size > 0:
            absolute_idxs = (search_start + micro_peaks).astype(int)
            # Añadimos como lista de índices (podés cambiar por dicts si querés amplitudes)
            for idx in absolute_idxs:
                ie_events.append(int(idx))

    # Eliminamos duplicados y ordenamos
    ie_events = sorted(set(ie_events))
    return ie_events


# ==========================================
# BLOQUE 3: Interfaz de Usuario
# ==========================================
def main():
    st.title("🩺 Detección de Asincronías: Fase 2 (DT) + Fase 3 (IE)")
    st.markdown("""
    Este módulo analiza formas de onda capturadas del ventilador para detectar **Doble Disparo (DT)** 
    y **Esfuerzos Inefectivos (IE)**. Capture una imagen clara donde la curva sea visible.
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

                # Extraemos el perfil central heurístico
                for col in range(start_col, end_col):
                    column_data = gray[:, col]
                    max_idx = int(np.argmax(column_data))
                    y_val = height - max_idx
                    signal_extracted.append(y_val)

                signal_np = np.asarray(signal_extracted, dtype=float)

            # --- Análisis Fase 2 (Doble Disparo) ---
            analysis = analyze_double_trigger(signal_np, sample_rate=fs_estimada, sensitivity=sensibilidad)

            # --- Análisis Fase 3 (Esfuerzos Inefectivos) ---
            ie_peaks = analyze_ineffective_efforts(analysis["signal_processed"], analysis["peaks"], sample_rate=fs_estimada)

            # --- Visualización de Resultados ---
            st.divider()
            st.subheader("Resultados del Análisis Multimodal")

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
            if analysis["signal_processed"] is not None:
                ax.plot(analysis["signal_processed"], label='Presión/Flujo', linewidth=1.5)
            else:
                ax.plot(signal_np, label='Presión/Flujo (raw)', linewidth=1.5)

            # 2. Picos Principales (Respiraciones)
            peaks_x = np.asarray(analysis["peaks"], dtype=int)
            if peaks_x.size > 0:
                peaks_y = (analysis["signal_processed"][peaks_x] if analysis["signal_processed"] is not None
                           else signal_np[peaks_x])
                ax.scatter(peaks_x, peaks_y, s=60, label='Disparo Ventilador', zorder=5)

            # 3. Doble Disparo (Líneas Rojas)
            if analysis["detected"]:
                for event in analysis["events"]:
                    p1, p2 = event["peak1"], event["peak2"]
                    y1 = analysis["signal_processed"][p1]
                    y2 = analysis["signal_processed"][p2]
                    ax.plot([p1, p2], [y1, y2], linewidth=3, linestyle='--')
                    ax.text(p2, y2 + 0.02 * (np.max(analysis["signal_processed"]) - np.min(analysis["signal_processed"])),
                            "DT", color='red', fontsize=9, fontweight='bold')

            # 4. Esfuerzos Inefectivos (Marcadores Naranjas)
            if len(ie_peaks) > 0:
                ie_idx = np.asarray(ie_peaks, dtype=int)
                ie_y = analysis["signal_processed"][ie_idx]
                ax.scatter(ie_idx, ie_y, marker='x', s=80, label='Esfuerzo Inefectivo', zorder=6)
                for idx in ie_idx:
                    ax.text(idx, analysis["signal_processed"][idx] + 0.02 * (np.max(analysis["signal_processed"]) - np.min(analysis["signal_processed"])),
                            "IE", color='orange', fontsize=8, ha='center')

            ax.set_title("Análisis de Asincronías: DT + IE")
            ax.set_xlabel("Tiempo (muestras)")
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

            # Mensajes finales
            if analysis["detected"]:
                st.warning("""
                ⚠️ **Alerta:** Se han detectado eventos compatibles con Doble Disparo. 
                Revise tiempos inspiratorios y ajustes de ciclado.
                """)
            else:
                st.success("Análisis completado: No se detectaron asincronías mayores.")

        else:
            st.error("Error: No se pudo decodificar la imagen.")
    else:
        st.info("Esperando captura de imagen...")


if __name__ == "__main__":
    main()
