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
        return {"rr": 0, "cycle_time": 0}
    
    # Tiempo total analizado en segundos
    total_time_sec = signal_len / sample_rate
    
    # Frecuencia Respiratoria (Respiraciones por Minuto)
    # Extrapolamos la cantidad de picos al minuto completo
    if total_time_sec > 0:
        rr = (len(peaks) / total_time_sec) * 60
    else:
        rr = 0
        
    return {"rr": int(rr), "duration": total_time_sec}

def analyze_flow_starvation(signal, peaks, sample_rate):
    """
    Detecta Hambre de Flujo (Flow Starvation) analizando la convexidad 
    de la curva de PRESIÓN durante la inspiración.
    """
    starvation_events =
    
    # Solo analizamos si hay picos claros
    if len(peaks) < 1:
        return starvation_events

    for p_idx in peaks:
        # Definimos la ventana de inspiración (asumimos que sube antes del pico)
        # Miramos 0.5 segundos hacia atrás desde el pico
        lookback = int(0.5 * sample_rate)
        start_insp = max(0, p_idx - lookback)
        
        # Segmento inspiratorio (subida)
        segment = signal[start_insp:p_idx]
        
        if len(segment) < 5:
            continue
            
        # --- Algoritmo de Convexidad ---
        # 1. Creamos una línea recta ideal desde el inicio hasta el pico
        x_seg = np.arange(len(segment))
        y_start = segment
        y_end = segment[-1]
        
        # Línea recta teórica (y = mx + b)
        slope = (y_end - y_start) / len(segment)
        ideal_line = slope * x_seg + y_start
        
        # 2. Calculamos la diferencia (Área bajo la curva vs línea ideal)
        # Si la señal real está muy por DEBAJO de la línea ideal, es una concavidad (Scooping)
        diff = ideal_line - segment
        max_concavity = np.max(diff)
        
        # Umbral heurístico: Si la concavidad es profunda (hambre de flujo)
        # Normalizamos respecto a la altura del pico para hacerlo independiente de la escala
        peak_height = y_end - np.min(signal)
        if peak_height > 0:
            normalized_concavity = max_concavity / peak_height
            
            # Si hay una "muesca" mayor al 15% de la altura del pico -> Alerta
            if normalized_concavity > 0.15: 
                starvation_events.append(p_idx - int(len(segment)/2)) # Marcamos la mitad de la subida

    return starvation_events

def analyze_double_trigger(signal_data, sample_rate=50, sensitivity=0.5):
    """Detecta Doble Disparo (Fase 2)."""
    signal = np.asarray(signal_data, dtype=float)
    results = {
        "detected": False, "event_count": 0, "events":, 
        "peaks":, "signal_processed": None, "message": ""
    }

    if signal.size == 0: return results

    # Suavizado
    try:
        window = 11
        poly = 3
        smoothed = savgol_filter(signal, window_length=window, polyorder=poly)
    except:
        smoothed = signal.copy()
    
    results["signal_processed"] = smoothed

    # Normalización (0 a 1) para consistencia en umbrales
    sig_min, sig_max = np.min(smoothed), np.max(smoothed)
    if sig_max - sig_min == 0: return results
    norm_sig = (smoothed - sig_min) / (sig_max - sig_min)

    # Detección de Picos
    prominence_val = max(0.05, 0.6 - (sensitivity * 0.5))
    min_dist = max(1, int(0.15 * sample_rate)) # 150ms refractario
    
    peaks, _ = find_peaks(norm_sig, prominence=prominence_val, distance=min_dist)
    results["peaks"] = peaks.tolist()

    # Lógica DT
    dt_thresh_sec = 0.8 # Umbral temporal para considerar "Doble"
    dt_events =
    
    if len(peaks) >= 2:
        for i in range(len(peaks) - 1):
            t_diff = (peaks[i+1] - peaks[i]) / sample_rate
            if t_diff < dt_thresh_sec:
                dt_events.append({
                    "peak1": peaks[i],
                    "peak2": peaks[i+1],
                    "time_diff": t_diff
                })

    results["events"] = dt_events
    results["event_count"] = len(dt_events)
    results["detected"] = len(dt_events) > 0
    return results

def analyze_ineffective_efforts(signal_data, major_peaks, sample_rate=50):
    """Detecta Esfuerzos Inefectivos (Fase 3)."""
    ie_events =
    if len(major_peaks) < 2: return ie_events
    
    for i in range(len(major_peaks) - 1):
        start = major_peaks[i]
        end = major_peaks[i+1]
        
        # Zona de búsqueda: Exhalación (evitamos el inicio y fin inmediatos)
        interval = end - start
        s_zone = start + int(interval * 0.25)
        e_zone = end - int(interval * 0.15)
        
        if e_zone <= s_zone: continue
        
        segment = signal_data[s_zone:e_zone]
        
        # Buscamos "micro-picos" con baja prominencia
        micro_peaks, _ = find_peaks(segment, prominence=0.02, width=3)
        
        for mp in micro_peaks:
            ie_events.append(s_zone + mp)
            
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
        curve_type = st.selectbox("Tipo de Curva Analizada", 
                                ["Flujo (Flow)", "Presión (Pressure/Paw)"],
                                help="Seleccione qué curva aparece en la foto para activar algoritmos específicos.")
        
        st.divider()
        st.header("Ajuste Algorítmico")
        sensibilidad = st.slider("Sensibilidad General", 0.0, 1.0, 0.5)
        fs_estimada = st.number_input("Escala de Tiempo (px/seg estimados)", 10, 200, 50, 
                                     help="Ajuste esto si los BPM calculados son irreales.")

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
            signal_raw =
            # Escaneo central (80% del ancho)
            for col in range(int(w*0.1), int(w*0.9)):
                # Buscamos el píxel más brillante/oscuro según contraste
                col_data = gray[:, col]
                # Asumimos curva clara sobre fondo oscuro
                y_val = h - np.argmax(col_data)
                signal_raw.append(y_val)
            
            signal_np = np.array(signal_raw)

            # --- PIPELINE DE ANÁLISIS ---
            
            # 1. Detección Base (Fase 2)
            analysis = analyze_double_trigger(signal_np, fs_estimada, sensibilidad)
            processed_sig = analysis["signal_processed"]
            major_peaks = analysis["peaks"]
            
            # 2. Detección Contextual (Fase 3 & 4)
            ie_events =
            starvation_events =
            
            # Solo buscamos IE si es Flujo (muescas en exhalación)
            if "Flujo" in curve_type:
                ie_events = analyze_ineffective_efforts(processed_sig, major_peaks, fs_estimada)
            
            # Solo buscamos Flow Starvation si es Presión (concavidad en inspiración)
            if "Presión" in curve_type:
                starvation_events = analyze_flow_starvation(processed_sig, major_peaks, fs_estimada)

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
                       delta="-Fatiga" if len(ie_events)>0 else "Ok", delta_color="inverse")
            kpi4.metric("Hambre de Flujo", len(starvation_events), 
                       delta="-Asincronía" if len(starvation_events)>0 else "Ok", delta_color="inverse")

            # Gráfico Maestro
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(processed_sig, color='cyan' if "Flujo" in curve_type else 'yellow', 
                   label=f'Curva de {curve_type}', linewidth=2)
            
            # Fondo oscuro estilo monitor médico
            ax.set_facecolor('#1e1e1e')
            fig.patch.set_facecolor('#0e1117')
            ax.grid(True, color='gray', alpha=0.2)
            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='y', colors='white')
            
            # Marcadores
            # Picos principales
            if major_peaks:
                ax.scatter(major_peaks, processed_sig[major_peaks], color='white', s=80, zorder=5, label='Trigger')

            # Doble Disparo (Rojo)
            for evt in analysis["events"]:
                p1, p2 = evt["peak1"], evt["peak2"]
                ax.plot([p1, p2], [processed_sig[p1], processed_sig[p2]], color='red', linewidth=4, linestyle=':')
                ax.text(p2, processed_sig[p2]+10, "DT", color='red', fontweight='bold')

            # Esfuerzos Inefectivos (Naranja) - Solo Flujo
            if ie_events:
                y_ie = processed_sig[ie_events]
                ax.scatter(ie_events, y_ie, color='orange', marker='x', s=100, linewidth=3, label='Esfuerzo Inefectivo')

            # Hambre de Flujo (Magenta) - Solo Presión
            if starvation_events:
                y_st = processed_sig[starvation_events]
                ax.scatter(starvation_events, y_st, color='magenta', marker='v', s=120, label='Hambre de Flujo')
                for st_idx in starvation_events:
                    ax.text(st_idx, processed_sig[st_idx]-20, "Flow\nStarvation", color='magenta', ha='center', fontsize=8)

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

if __name__ == "__main__":
    main()
