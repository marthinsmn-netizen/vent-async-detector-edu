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
    """
    Calcula métricas clínicas básicas como la Frecuencia Respiratoria (RR).
    """
    if len(peaks) < 2:
        return {"rr": 0, "cycle_time": 0}
    
    # Tiempo total analizado en segundos
    total_time_sec = float(signal_len) / float(sample_rate)
    
    # Frecuencia Respiratoria (Respiraciones por Minuto)
    # Extrapolamos la cantidad de picos al minuto completo
    if total_time_sec > 0:
        rr = (len(peaks) / total_time_sec) * 60.0
    else:
        rr = 0.0
        
    return {"rr": int(round(rr)), "duration": total_time_sec}

def analyze_flow_starvation(signal, peaks, sample_rate):
    """
    Detecta Hambre de Flujo (Flow Starvation) analizando la convexidad 
    de la curva de PRESIÓN durante la fase de subida (inspiración).
    """
    starvation_events = list()
    
    # Validaciones de seguridad
    signal = np.asarray(signal, dtype=float)
    if signal.size == 0 or len(peaks) < 1:
        return starvation_events

    for p_idx in peaks:
        p_idx = int(p_idx)
        
        # Definimos la ventana de inspiración: miramos hacia atrás desde el pico.
        # Asumimos que la subida ocurre en los 0.5 segundos previos aprox.
        lookback = int(0.5 * sample_rate)
        start_insp = max(0, p_idx - lookback)
        
        # Segmento inspiratorio (la rampa de subida)
        segment = signal[start_insp:p_idx]
        
        # Si el segmento es muy corto (ruido), lo ignoramos
        if segment.size < 5:
            continue
            
        # --- Algoritmo de Convexidad (Scooping Analysis) ---
        
        # 1. Línea de Referencia Ideal: Conectamos el inicio con el pico.
        # Si la respiración fuera perfectamente pasiva, sería casi recta o convexa.
        x_seg = np.arange(len(segment))
        y_start = float(segment)
        y_end = float(segment[-1])
        
        # Ecuación de la recta: y = mx + b
        if len(segment) > 1:
            slope = (y_end - y_start) / (len(segment) - 1)
        else:
            slope = 0
            
        ideal_line = slope * x_seg + y_start
        
        # 2. Cálculo del Déficit (Área bajo la línea ideal)
        # "diff" positivo significa que la señal real está POR DEBAJO de la ideal (cóncava)
        diff = ideal_line - segment
        max_concavity = float(np.max(diff))
        
        # 3. Normalización
        # Hacemos el umbral relativo a la altura de la respiración (para que funcione a cualquier escala)
        peak_height = y_end - float(np.min(signal))
        
        if peak_height > 0:
            normalized_concavity = max_concavity / peak_height
            
            # UMBRAL CLÍNICO: 
            # Si la "panza" de la curva es mayor al 15% de la altura total, es sospechoso.
            if normalized_concavity > 0.15: 
                # Guardamos el punto medio de la subida para marcarlo en el gráfico
                mark_idx = start_insp + int(len(segment)/2)
                starvation_events.append(mark_idx)

    # Limpieza de duplicados
    return sorted(list(set(starvation_events)))

def analyze_double_trigger(signal_data, sample_rate=50, sensitivity=0.5):
    """Detecta Doble Disparo (Fase 2)."""
    signal = np.asarray(signal_data, dtype=float)
    results = {
        "detected": False,
        "event_count": 0,
        "events": list(), 
        "peaks": list(), 
        "signal_processed": None, 
        "message": ""
    }

    if signal.size == 0:
        results["message"] = "Señal vacía."
        return results

    # Suavizado
    try:
        window = 11
        if window >= signal.size:
            window = max(3, signal.size - 2 if signal.size % 2 == 0 else signal.size - 1)
        poly = 3
        smoothed = savgol_filter(signal, window_length=window, polyorder=min(poly, window-1))
    except:
        smoothed = signal.copy()
    
    results["signal_processed"] = smoothed

    # Normalización
    sig_min, sig_max = float(np.min(smoothed)), float(np.max(smoothed))
    if sig_max - sig_min == 0:
        results["message"] = "Señal plana."
        return results
        
    norm_sig = (smoothed - sig_min) / (sig_max - sig_min)

    # Detección de Picos
    prominence_val = max(0.05, 0.6 - (sensitivity * 0.5))
    min_dist = max(1, int(0.15 * sample_rate)) 
    
    peaks, _ = find_peaks(norm_sig, prominence=prominence_val, distance=min_dist)
    peaks = np.asarray(peaks, dtype=int)
    results["peaks"] = peaks.tolist()

    # Lógica DT
    dt_thresh_sec = 0.8 
    dt_events = list()
    
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
    """Detecta Esfuerzos Inefectivos (Fase 3)."""
    ie_events = list()
    signal = np.asarray(signal_data, dtype=float)
    major_peaks_arr = np.asarray(major_peaks, dtype=int)
    
    if major_peaks_arr.size < 2: return ie_events
    
    for i in range(major_peaks_arr.size - 1):
        start = int(major_peaks_arr[i])
        end = int(major_peaks_arr[i+1])
        
        # Zona de búsqueda: Exhalación (el valle entre picos)
        interval = end - start
        if interval < 5: continue
        
        s_zone = start + int(interval * 0.25)
        e_zone = end - int(interval * 0.15)
        
        if e_zone <= s_zone: continue
        
        segment = signal[s_zone:e_zone]
        if segment.size == 0: continue
        
        # Buscamos "micro-picos"
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
    
    Este sistema detecta anomalías basándose en la morfología de la onda.
    Seleccione el tipo de curva correcta para habilitar los algoritmos específicos.
    """)
    
    # --- Sidebar de Configuración ---
    with st.sidebar:
        st.header("Configuración Clínica")
        
        # SELECTOR DE CONTEXTO (CRÍTICO PARA FASE 4)
        curve_type = st.selectbox(
            "¿Qué curva estás analizando?", 
            ["Flujo (Flow)", "Presión (Pressure/Paw)"],
            index=0,
            help="El Hambre de Flujo se busca en Presión. Los Esfuerzos Inefectivos se ven mejor en Flujo."
        )
        
        st.divider()
        st.header("Ajuste Fino")
        sensibilidad = st.slider("Sensibilidad", 0.0, 1.0, 0.5)
        fs_estimada = int(st.number_input("Escala (px/seg estimados)", 10, 200, 50))

    # --- Entrada de Datos ---
    img_buffer = st.camera_input("📸 Capturar Pantalla")

    if img_buffer:
        bytes_data = img_buffer.getvalue()
        img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        
        if img is not None:
            st.image(img, caption="Imagen Original", use_column_width=True)
            
            # Extracción de Señal
            with st.spinner("Digitalizando curva..."):
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                h, w = gray.shape
                signal_raw = list()
                
                # Escaneo central del 80%
                start_col = int(w*0.1)
                end_col = int(w*0.9)
                
                for col in range(start_col, end_col):
                    col_data = gray[:, col]
                    # Invertimos Y porque en imágenes 0 está arriba
                    y_val = int(h - np.argmax(col_data))
                    signal_raw.append(y_val)
                
                signal_np = np.array(signal_raw)

            # --- ORQUESTADOR DE ANÁLISIS ---
            
            # 1. Análisis Base (Común a todo)
            analysis = analyze_double_trigger(signal_np, fs_estimada, sensibilidad)
            processed_sig = analysis["signal_processed"]
            major_peaks = analysis["peaks"]
            
            # 2. Análisis Contextual (Depende del Selectbox)
            ie_events = list()
            starvation_events = list()
            
            if "Flujo" in curve_type:
                # En Flujo buscamos muescas en la exhalación (Fase 3)
                ie_events = analyze_ineffective_efforts(processed_sig, major_peaks, fs_estimada)
            
            elif "Presión" in curve_type:
                # En Presión buscamos concavidad en la inspiración (Fase 4)
                starvation_events = analyze_flow_starvation(processed_sig, major_peaks, fs_estimada)

            # 3. Métricas
            metrics = analyze_clinical_metrics(major_peaks, len(signal_np), fs_estimada)

            # --- VISUALIZACIÓN (DASHBOARD) ---
            st.divider()
            
            # Tarjetas de Métricas (KPIs)
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Frecuencia (RPM)", f"{metrics['rr']} rpm", 
                     help="Calculado en base a la escala temporal estimada")
            
            k2.metric("Doble Disparo", analysis["event_count"], 
                     delta="-Riesgo" if analysis["detected"] else "Normal", delta_color="inverse")
            
            k3.metric("Esfuerzos Inefectivos", len(ie_events), 
                     delta="-Fatiga" if len(ie_events)>0 else "Normal", delta_color="inverse")
            
            k4.metric("Hambre de Flujo", len(starvation_events), 
                     delta="-Asincronía" if len(starvation_events)>0 else "Normal", delta_color="inverse")

            # Gráfico Principal
            fig, ax = plt.subplots(figsize=(12, 5))
            
            # Estilo "Monitor Médico" (Fondo oscuro)
            fig.patch.set_facecolor('#0e1117')
            ax.set_facecolor('#1e1e1e')
            
            # Color de línea según tipo (Amarillo=Presión, Cian=Flujo)
            line_color = 'yellow' if "Presión" in curve_type else 'cyan'
            
            ax.plot(processed_sig, color=line_color, linewidth=2, label=f'Curva de {curve_type}')
            
            # Marcadores de Eventos
            if len(major_peaks) > 0:
                ax.scatter(major_peaks, processed_sig[major_peaks], color='white', s=30, label='Ciclo', zorder=5)

            # Doble Disparo (Rojo)
            for evt in analysis["events"]:
                p1, p2 = evt["peak1"], evt["peak2"]
                ax.plot([p1, p2], [processed_sig[p1], processed_sig[p2]], color='red', linewidth=3, linestyle='--')
                ax.text(p2, processed_sig[p2]+10, "DT", color='red', fontsize=12, fontweight='bold')

            # Esfuerzos Inefectivos (Naranja - Solo Flujo)
            if len(ie_events) > 0:
                y_ie = processed_sig[ie_events]
                ax.scatter(ie_events, y_ie, color='orange', marker='x', s=100, linewidth=3, label='Esfuerzo Inefectivo')

            # Hambre de Flujo (Magenta - Solo Presión)
            if len(starvation_events) > 0:
                y_st = processed_sig[starvation_events]
                ax.scatter(starvation_events, y_st, color='magenta', marker='v', s=100, label='Flow Starvation')
                for st_idx in starvation_events:
                    ax.text(st_idx, processed_sig[st_idx]-15, "FS", color='magenta', ha='center', fontsize=10)

            # Configuración de Ejes
            ax.grid(True, color='gray', alpha=0.2)
            ax.tick_params(colors='white')
            for spine in ax.spines.values(): spine.set_edgecolor('gray')
            
            leg = ax.legend(facecolor='#1e1e1e', edgecolor='gray')
            plt.setp(leg.get_texts(), color='white')
            
            st.pyplot(fig)
            
            # Mensajes Clínicos
            if len(starvation_events) > 0:
                st.info("💡 **Consejo Clínico:** Se detectó Hambre de Flujo (concavidad inspiratoria). Considere aumentar el flujo inspiratorio o cambiar el Rise Time.")
            
            if analysis["detected"]:
                st.error("🚨 **Alerta:** Doble Disparo detectado. Posible causa: Tiempo Inspiratorio neural > Tiempo mecánico.")

        else:
            st.error("No se pudo procesar la imagen.")

if __name__ == "__main__":
    main()
