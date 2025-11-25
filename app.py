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
    """Calcula métricas clínicas básicas (Frecuencia Respiratoria)."""
    if len(peaks) < 2:
        return {"rr": 0, "duration": 0}
    
    # Tiempo total analizado en segundos
    total_time_sec = float(signal_len) / float(sample_rate) if sample_rate > 0 else 0.0
    
    # Frecuencia Respiratoria (Respiraciones por Minuto)
    if total_time_sec > 0:
        rr = (len(peaks) / total_time_sec) * 60.0
    else:
        rr = 0.0
        
    return {"rr": int(round(rr)), "duration": total_time_sec}


def analyze_flow_starvation(signal, peaks, sample_rate):
    """
    Detecta Hambre de Flujo (Flow Starvation) analizando la convexidad 
    de la curva de PRESIÓN durante la inspiración.
    """
    starvation_events = []  # inicialización corregida

    # Validaciones básicas
    signal = np.asarray(signal, dtype=float)
    if signal.size == 0 or len(peaks) < 1:
        return starvation_events

    for p_idx in peaks:
        p_idx = int(p_idx)
        # Definimos la ventana de inspiración (asumimos que sube antes del pico)
        lookback = int(0.5 * sample_rate)
        start_insp = max(0, p_idx - lookback)
        
        # Segmento inspiratorio (subida)
        segment = signal[start_insp:p_idx]
        
        if segment.size < 5:
            continue
            
        # --- Algoritmo de Convexidad ---
        # 1. Creamos una línea recta ideal desde el inicio hasta el pico usando valores reales
        x_seg = np.arange(len(segment))
        y_start = float(segment[0])
        y_end = float(segment[-1])
        
        # Línea recta teórica (y = m*x + b)
        # Usamos len(segment)-1 para evitar división por cero y para que la recta vaya del primer al último punto
        denom = max(1, (len(segment) - 1))
        slope = (y_end - y_start) / denom
        ideal_line = slope * x_seg + y_start
        
        # 2. Calculamos la diferencia (línea ideal - señal real)
        diff = ideal_line - segment
        max_concavity = float(np.max(diff))
        
        # Normalizamos respecto a la altura del pico global para independencia de escala
        peak_height = y_end - float(np.min(signal))
        if peak_height <= 0:
            continue

        normalized_concavity = max_concavity / peak_height
        
        # Umbral heurístico de concavidad
        if normalized_concavity > 0.15:
            # Marcamos un índice representativo dentro de la subida
            mark_idx = start_insp + int(len(segment) / 2)
            starvation_events.append(int(mark_idx))

    # Ordenar y quitar duplicados
    starvation_events = sorted(set(starvation_events))
    return starvation_events


def analyze_double_trigger(signal_data, sample_rate=50, sensitivity=0.5):
    """Detecta Doble Disparo (Fase 2)."""
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

    # Suavizado (ajustando ventana si la señal es corta)
    try:
        window = 11
        if window >= signal.size:
            # ventana impar < tamaño de señal
            window = signal.size - 1 if (signal.size - 1) % 2 == 1 else signal.size - 2
            window = max(3, int(window))
        poly = 3
        smoothed = savgol_filter(signal, window_length=window, polyorder=min(poly, window-1))
    except Exception:
        smoothed = signal.copy()
    
    results["signal_processed"] = smoothed

    # Normalización (0 a 1)
    sig_min, sig_max = float(np.min(smoothed)), float(np.max(smoothed))
    if sig_max - sig_min == 0:
        results["message"] = "Señal plana o sin variación detectada."
        return results
    norm_sig = (smoothed - sig_min) / (sig_max - sig_min)

    # Detección de Picos
    prominence_val = max(0.05, 0.6 - (sensitivity * 0.5))
    min_dist = max(1, int(0.15 * sample_rate))  # 150 ms refractario aproximado
    
    peaks, _ = find_peaks(norm_sig, prominence=prominence_val, distance=min_dist)
    peaks = np.asarray(peaks, dtype=int)
    results["peaks"] = peaks.tolist()

    # Lógica DT
    dt_thresh_sec = 0.8  # Umbral temporal para considerar "Doble"
    dt_events = []       # inicialización corregida
    
    if peaks.size >= 2:
        for i in range(peaks.size - 1):
            t_diff = float(peaks[i+1] - peaks[i]) / float(sample_rate)
            if 0 < t_diff < dt_thresh_sec:
                dt_events.append({
                    "peak1": int(peaks[i]),
                    "peak2": int(peaks[i+1]),
                    "time_diff": float(t_diff)
                })

    results["events"] = dt_events
    results["event_count"] = len(dt_events)
    results["detected"] = len(dt_events) > 0
    return results


def analyze_ineffective_efforts(signal_data, major_peaks, sample_rate=50):
    """Detecta Esfuerzos Inefectivos (Fase 3)."""
    ie_events = []  # inicialización corregida
    signal = np.asarray(signal_data, dtype=float)

    major_peaks_arr = np.asarray(major_peaks, dtype=int)
    if major_peaks_arr.size < 2 or signal.size == 0:
        return ie_events
    
    for i in range(major_peaks_arr.size - 1):
        start = int(major_peaks_arr[i])
        end = int(major_peaks_arr[i+1])
        
        # Zona de búsqueda: Exhalación (evitamos el inicio y fin inmediatos)
        interval = end - start
        if interval <= 3:
            continue

        s_zone = start + int(interval * 0.25)
        e_zone = end - int(interval * 0.15)
        
        if e_zone <= s_zone:
            continue
        
        segment = signal[s_zone:e_zone]
        if segment.size == 0:
            continue
        
        # Buscamos "micro-picos" con baja prominencia
        micro_peaks, _ = find_peaks(segment, prominence=0.02, width=3)
        
        for mp in micro_peaks:
            ie_events.append(int(s_zone + int(mp)))
            
    # Deduplicar y ordenar
    return sorted(list(set(ie_events)))

# ==========================================
# BLOQUE 3: Interfaz de Usuario (UI)
# ==========================================

def main():
    st.title("🫁 Ventilator Lab: Análisis de Asincronías")
    st.markdown("### Fase 4: Detección Multi-Modo (DT + IE + Flow Starvation)")
    
    # --- Sidebar de Configuración ---
    with st.sidebar:
        st.header("Parámetros Clínicos")
        curve_type = st.selectbox(
            "Tipo de Curva Analizada", 
            ["Flujo (Flow)", "Presión (Pressure/Paw)"],
            help="Seleccione qué curva aparece en la foto para activar algoritmos específicos."
        )
        
        st.divider()
        st.header("Ajuste Algorítmico")
        sensibilidad = st.slider("Sensibilidad General", 0.0, 1.0, 0.5)
        fs_estimada = int(st.number_input(
            "Escala de Tiempo (px/seg estimados)", 10, 200, 50,
            help="Ajuste esto si los BPM calculados son irreales."
        ))

    # --- Entrada de Datos ---
    img_buffer = st.camera_input("📸 Capturar Pantalla del Ventilador")

    if img_buffer:
        # Procesamiento de Imagen
        bytes_data = img_buffer.getvalue()
        img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        
        if img is not None:
            st.image(img, caption="Imagen Original", use_column_width=True)
            
            # Extracción de Señal (Heurística de brillo)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            signal_raw = []  # inicialización corregida
            # Escaneo central (80% del ancho)
            start_col = int(w * 0.1)
            end_col = int(w * 0.9)
            for col in range(start_col, end_col):
                col_data = gray[:, col]
                # Asumimos curva clara sobre fondo oscuro; si es al revés cambiar lógica
                y_val = int(h - np.argmax(col_data))
                signal_raw.append(y_val)
            
            signal_np = np.asarray(signal_raw, dtype=float)
            if signal_np.size == 0:
                st.error("No se pudo extraer una señal válida de la imagen.")
                return

            # --- PIPELINE DE ANÁLISIS ---
            
            # 1. Detección Base (Fase 2)
            analysis = analyze_double_trigger(signal_np, sample_rate=fs_estimada, sensitivity=sensibilidad)
            processed_sig = analysis["signal_processed"]
            major_peaks = analysis["peaks"]
            
            # 2. Detección Contextual (Fase 3 & 4)
            ie_events = []           # inicialización corregida
            starvation_events = []   # inicialización corregida
            
            # Solo buscamos IE si es Flujo (muescas en exhalación)
            if "Flujo" in curve_type:
                ie_events = analyze_ineffective_efforts(processed_sig if processed_sig is not None else signal_np,
                                                       major_peaks, sample_rate=fs_estimada)
            
            # Solo buscamos Flow Starvation si es Presión (concavidad en inspiración)
            if "Presión" in curve_type:
                starvation_events = analyze_flow_starvation(processed_sig if processed_sig is not None else signal_np,
                                                            major_peaks, sample_rate=fs_estimada)

            # 3. Métricas Clínicas
            metrics = analyze_clinical_metrics(major_peaks, len(signal_np), fs_estimada)

            # --- DASHBOARD DE RESULTADOS ---
            st.divider()
            
            # KPIs Clínicos
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric("Frecuencia (RPM)", f"{metrics['rr']} rpm", help="Respiraciones por minuto estimadas")
            kpi2.metric("Doble Disparo", analysis["event_count"], 
                       delta="-Riesgo VILI" if analysis["detected"] else "Ok", delta_color="inverse")
            kpi3.metric("Esfuerzos Inefectivos", len(ie_events), 
                       delta="-Fatiga" if len(ie_events) > 0 else "Ok", delta_color="inverse")
            kpi4.metric("Hambre de Flujo", len(starvation_events), 
                       delta="-Asincronía" if len(starvation_events) > 0 else "Ok", delta_color="inverse")

            # Gráfico Maestro
            fig, ax = plt.subplots(figsize=(12, 5))
            plot_sig = processed_sig if processed_sig is not None else signal_np
            color = 'cyan' if "Flujo" in curve_type else 'yellow'
            ax.plot(plot_sig, color=color, label=f'Curva de {curve_type}', linewidth=2)
            
            # Fondo oscuro estilo monitor médico
            ax.set_facecolor('#1e1e1e')
            fig.patch.set_facecolor('#0e1117')
            ax.grid(True, color='gray', alpha=0.2)
            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='y', colors='white')
            
            # Marcadores
            # Picos principales
            if len(major_peaks) > 0:
                peaks_idx = np.asarray(major_peaks, dtype=int)
                peaks_y = plot_sig[peaks_idx]
                ax.scatter(peaks_idx, peaks_y, color='white', s=80, zorder=5, label='Trigger')

            # Doble Disparo (Rojo)
            for evt in analysis["events"]:
                p1, p2 = int(evt["peak1"]), int(evt["peak2"])
                # Protección por rango
                if p1 < 0 or p2 >= plot_sig.size: 
                    continue
                ax.plot([p1, p2], [plot_sig[p1], plot_sig[p2]], color='red', linewidth=4, linestyle=':')
                ax.text(p2, plot_sig[p2] + 10, "DT", color='red', fontweight='bold')

            # Esfuerzos Inefectivos (Naranja) - Solo Flujo
            if len(ie_events) > 0:
                ie_idx = np.asarray(ie_events, dtype=int)
                ie_idx = ie_idx[(ie_idx >= 0) & (ie_idx < plot_sig.size)]
                if ie_idx.size > 0:
                    y_ie = plot_sig[ie_idx]
                    ax.scatter(ie_idx, y_ie, color='orange', marker='x', s=100, linewidth=3, label='Esfuerzo Inefectivo')

            # Hambre de Flujo (Magenta) - Solo Presión
            if len(starvation_events) > 0:
                st_idx = np.asarray(starvation_events, dtype=int)
                st_idx = st_idx[(st_idx >= 0) & (st_idx < plot_sig.size)]
                if st_idx.size > 0:
                    y_st = plot_sig[st_idx]
                    ax.scatter(st_idx, y_st, color='magenta', marker='v', s=120, label='Hambre de Flujo')
                    for s_i in st_idx:
                        ax.text(int(s_i), plot_sig[int(s_i)] - 20, "Flow\nStarvation", color='magenta', ha='center', fontsize=8)

            # Leyenda
            leg = ax.legend(facecolor='#1e1e1e', edgecolor='white')
            plt.setp(leg.get_texts(), color='white')
            
            st.pyplot(fig)
            
            # Recomendaciones Clínicas
            if len(starvation_events) > 0:
                st.info("💡 **Consejo Clínico:** Se detectó Hambre de Flujo. Considere aumentar el Flujo Inspiratorio o cambiar a modo Presión Soporte para satisfacer la demanda del paciente.")
            
            if analysis["detected"]:
                st.error("🚨 **Alerta Crítica:** Doble Disparo detectado. Riesgo de Volutrauma. Revise si el Tiempo Inspiratorio es demasiado corto.")

        else:
            st.error("Error procesando la imagen.")
    else:
        st.info("Esperando captura de imagen...")

if __name__ == "__main__":
    main()
