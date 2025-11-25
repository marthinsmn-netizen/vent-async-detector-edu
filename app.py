# ==========================================
# BLOQUE 1: Importaciones y Configuración
# ==========================================
import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter

# Configuración para entornos sin display (Streamlit Cloud)
import matplotlib
matplotlib.use('Agg')

st.set_page_config(
    page_title="Ventilador Lab AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# BLOQUE 2: Funciones Core de Procesamiento
# ==========================================

def analyze_clinical_metrics(peaks, signal_len, sample_rate):
    if len(peaks) < 2:
        return {"rr": 0, "cycle_time": 0}

    total_time_sec = float(signal_len) / float(sample_rate)
    rr = (len(peaks) / total_time_sec) * 60.0 if total_time_sec > 0 else 0.0
    return {"rr": int(round(rr)), "duration": total_time_sec}

def analyze_flow_starvation(signal, peaks, sample_rate):
    starvation_events = list()
    signal = np.asarray(signal, dtype=float)
    if signal.size == 0 or len(peaks) < 1:
        return starvation_events

    for p_idx in peaks:
        p_idx = int(p_idx)
        lookback = int(0.5 * sample_rate)
        start_insp = max(0, p_idx - lookback)
        segment = signal[start_insp:p_idx]

        if segment.size < 5:
            continue

        x_seg = np.arange(len(segment))
        y_start = float(segment[0])
        y_end = float(segment[-1])

        if len(segment) > 1:
            slope = (y_end - y_start) / (len(segment) - 1)
        else:
            slope = 0

        ideal_line = slope * x_seg + y_start
        diff = ideal_line - segment
        max_concavity = float(np.max(diff))

        peak_height = max(1e-6, y_end - float(np.min(signal)))
        normalized_concavity = max_concavity / peak_height

        if normalized_concavity > 0.15:
            mark_idx = start_insp + int(len(segment) / 2)
            starvation_events.append(mark_idx)

    return sorted(list(set(starvation_events)))

def analyze_double_trigger(signal_data, sample_rate=50, sensitivity=0.5):
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

    try:
        window = 11
        if window >= signal.size:
            window = max(3, signal.size - (2 if signal.size % 2 == 0 else 1))
        poly = 3
        smoothed = savgol_filter(signal, window_length=window, polyorder=min(poly, window-1))
    except:
        smoothed = signal.copy()

    results["signal_processed"] = smoothed

    sig_min, sig_max = float(np.min(smoothed)), float(np.max(smoothed))
    if sig_max - sig_min == 0:
        results["message"] = "Señal plana."
        return results

    norm_sig = (smoothed - sig_min) / (sig_max - sig_min)

    prominence_val = max(0.05, 0.6 - (sensitivity * 0.5))
    min_dist = max(1, int(0.15 * sample_rate))

    peaks, _ = find_peaks(norm_sig, prominence=prominence_val, distance=min_dist)
    peaks = np.asarray(peaks, dtype=int)
    results["peaks"] = peaks.tolist()

    dt_thresh_sec = 0.8
    dt_events = []

    if peaks.size >= 2:
        for i in range(peaks.size - 1):
            t_diff = float(peaks[i+1] - peaks[i]) / float(sample_rate)
            if 0 < t_diff < dt_thresh_sec:
                dt_events.append({
                    "peak1": int(peaks[i]),
                    "peak2": int(peaks[i+1]),
                    "time_diff": t_diff
                })

    results["events"] = dt_events
    results["event_count"] = len(dt_events)
    results["detected"] = len(dt_events) > 0
    return results

def analyze_ineffective_efforts(signal_data, major_peaks, sample_rate=50):
    ie_events = []
    signal = np.asarray(signal_data, dtype=float)
    major_peaks_arr = np.asarray(major_peaks, dtype=int)

    if major_peaks_arr.size < 2:
        return ie_events

    for i in range(major_peaks_arr.size - 1):
        start = int(major_peaks_arr[i])
        end = int(major_peaks_arr[i+1])

        interval = end - start
        if interval < 5:
            continue

        s_zone = start + int(interval * 0.25)
        e_zone = end - int(interval * 0.15)

        if e_zone <= s_zone:
            continue

        segment = signal[s_zone:e_zone]
        if segment.size == 0:
            continue

        micro_peaks, _ = find_peaks(segment, prominence=0.02, width=3)
        for mp in micro_peaks:
            ie_events.append(int(s_zone + mp))

    return sorted(list(set(ie_events)))

# ==========================================
# BLOQUE 3: Interfaz de Usuario (UI)
# ==========================================

def main():
    st.title("🫁 Ventilator Lab: Análisis Multi-Modo")
    st.markdown("""
    **Sistema de Detección de Asincronías Fase 4**
    """)

    with st.sidebar:
        st.header("Configuración Clínica")
        curve_type = st.selectbox(
            "¿Qué curva estás analizando?",
            ["Flujo (Flow)", "Presión (Pressure/Paw)"],
            index=0
        )
        st.divider()
        st.header("Ajuste Fino")
        sensibilidad = st.slider("Sensibilidad", 0.0, 1.0, 0.5)
        fs_estimada = int(st.number_input("Escala (px/seg estimados)", 10, 200, 50))

    img_buffer = st.camera_input("📸 Capturar Pantalla")

    if img_buffer:
        bytes_data = img_buffer.getvalue()
        img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        if img is not None:
            st.image(img, caption="Imagen Original", use_column_width=True)

            with st.spinner("Digitalizando curva..."):
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                h, w = gray.shape
                signal_raw = []

                start_col = int(w * 0.1)
                end_col = int(w * 0.9)

                for col in range(start_col, end_col):
                    col_data = gray[:, col]
                    y_val = int(h - np.argmax(col_data))
                    signal_raw.append(y_val)

                signal_np = np.array(signal_raw)

            analysis = analyze_double_trigger(signal_np, fs_estimada, sensibilidad)
            processed_sig = analysis["signal_processed"]
            major_peaks = analysis["peaks"]

            major_peaks = [p for p in major_peaks if 0 <= p < len(processed_sig)]

            ie_events = []
            starvation_events = []

            if "Flujo" in curve_type:
                ie_events = analyze_ineffective_efforts(processed_sig, major_peaks, fs_estimada)

            elif "Presión" in curve_type:
                starvation_events = analyze_flow_starvation(processed_sig, major_peaks, fs_estimada)

            metrics = analyze_clinical_metrics(major_peaks, len(signal_np), fs_estimada)

            st.divider()

            # KPIs
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Frecuencia (RPM)", f"{metrics['rr']} rpm")
            k2.metric("Doble Disparo", analysis["event_count"])
            k3.metric("Esfuerzos Inefectivos", len(ie_events))
            k4.metric("Hambre de Flujo", len(starvation_events))

            fig, ax = plt.subplots(figsize=(12, 5))
            fig.patch.set_facecolor('#0e1117')
            ax.set_facecolor('#1e1e1e')

            line_color = 'yellow' if "Presión" in curve_type else 'cyan'
            ax.plot(processed_sig, color=line_color, linewidth=2)

            if len(major_peaks) > 0:
                ax.scatter(
                    major_peaks,
                    processed_sig[major_peaks],
                    color='white', s=30
                )

            # Double trigger
            for evt in analysis["events"]:
                p1, p2 = evt["peak1"], evt["peak2"]
                if p1 < len(processed_sig) and p2 < len(processed_sig):
                    ax.plot(
                        [p1, p2],
                        [processed_sig[p1], processed_sig[p2]],
                        color='red',
                        linewidth=3,
                        linestyle='--'
                    )
                    ax.text(
                        p2,
                        processed_sig[p2] + 10,
                        "DT",
                        color='red',
                        fontsize=12,
                        fontweight='bold'
                    )

            # IE
            if len(ie_events) > 0:
                ie_events = [i for i in ie_events if 0 <= i < len(processed_sig)]
                y_ie = processed_sig[ie_events]
                ax.scatter(
                    ie_events,
                    y_ie,
                    color='orange',
                    marker='x',
                    s=100,
                    linewidth=3
                )

            # ==========================================
            # BLOQUE FS — CORRECTAMENTE INDENTADO
            # ==========================================
            if len(starvation_events) > 0:
                starvation_events = [
                    i for i in starvation_events
                    if 0 <= i < len(processed_sig)
                ]

                y_fs = processed_sig[starvation_events]

                ax.scatter(
                    starvation_events,
                    y_fs,
                    color='magenta',
                    marker='D',
                    s=90,
                    linewidth=2
                )

            st.pyplot(fig)

# Ejecutar
if __name__ == "__main__":
    main()
