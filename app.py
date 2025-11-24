# ==========================================
# BLOQUE 1: Importaciones y Configuración
# ==========================================
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter

# Configuración crítica para entornos sin display (Streamlit Cloud, Docker)
# Esto evita errores de Tcl/Tk o X11 al generar gráficos.
import matplotlib
matplotlib.use('Agg') 

# Configuración de página (Opcional, pero recomendado)
st.set_page_config(
    page_title="Detector de Asincronías Ventilatorias",
    layout="wide",
    initial_sidebar_state="expanded"
)
# ==========================================
# BLOQUE 2: Funciones Core de Procesamiento (Fase 2)
# ==========================================

def analyze_double_trigger(signal_data, sample_rate=50, sensitivity=0.5):
    """
    Detecta eventos de Doble Disparo en una señal unidimensional de ventilación.
    
    Args:
        signal_data (np.array): Vector 1D con los valores de amplitud (Flujo/Presión).
        sample_rate (int): Frecuencia de muestreo estimada (Hz). Por defecto 50.
                           En imágenes, esto depende de la escala temporal horizontal.
        sensitivity (float): Ajuste fino para la detección de picos (0.0 a 1.0).
                             Mayor sensibilidad = detecta picos más pequeños (más falsos positivos).
    
    Returns:
        dict: Resultados incluyendo índices de picos, eventos DT, señal suavizada y métricas.
    """
    results = {
       results == {
        "detected": False,
        "event_count": 0,
        "events": list(), 
        "peaks": list(),   
        "signal_processed": None,
        "message": ""
    }}

    # --- Paso 1: Preprocesamiento y Suavizado ---
    # Las señales extraídas de imágenes tienen ruido de cuantización.
    # Aplicamos filtro Savitzky-Golay que es superior a la media móvil para conservar picos.
    # window_length debe ser impar y polyorder < window_length.
    try:
        window = 11  # Ventana de suavizado
        poly = 3     # Orden del polinomio
        smoothed = savgol_filter(signal_data, window_length=window, polyorder=poly)
    except Exception as e:
        # Fallback si la señal es muy corta
        smoothed = signal_data
    
    results["signal_processed"] = smoothed

    # --- Paso 2: Normalización ---
    # Normalizamos la señal a rango  para usar umbrales de prominencia universales.
    sig_min, sig_max = np.min(smoothed), np.max(smoothed)
    if sig_max - sig_min == 0:
        results["message"] = "Señal plana o sin variación detectada."
        return results
        
    norm_sig = (smoothed - sig_min) / (sig_max - sig_min)

    # --- Paso 3: Configuración de Parámetros de Find_Peaks ---
    # Prominencia: Inversamente proporcional a la sensibilidad.
    # Sensibilidad 1.0 -> Prominencia 0.1 (detecta todo).
    # Sensibilidad 0.1 -> Prominencia 0.6 (solo picos muy grandes).
    prominence_val = max(0.1, 0.6 - (sensitivity * 0.5))
    
    # Distancia Mínima: Refractariedad fisiológica absoluta.
    # Incluso en DT, los picos no están pegados instantáneamente. Asumimos min 0.2s.
    min_dist_samples = int(0.2 * sample_rate)
    
    # Ancho Mínimo: Evita detectar ruido de "spike" (un solo punto alto).
    min_width_samples = int(0.05 * sample_rate)

    # Ejecución del algoritmo scipy.signal.find_peaks
    peaks, properties = find_peaks(
        norm_sig,
        prominence=prominence_val,
        distance=min_dist_samples,
        width=min_width_samples
    )
    results["peaks"] = peaks

    # --- Paso 4: Lógica de Detección de Doble Disparo ---
    # Definición: Dos ciclos separados por un tiempo espiratorio muy corto.
    # Umbral de tiempo crítico: < 0.8 segundos entre inicios de inspiración.
    dt_threshold_seconds = 1.0 
    dt_threshold_samples = dt_threshold_seconds * sample_rate
    
    dt_events = # Lista vacia corregida
    
    if len(peaks) >= 2:
        for i in range(len(peaks) - 1):
            idx_current = peaks[i]
            idx_next = peaks[i+1]
            
            interval_samples = idx_next - idx_current
            interval_seconds = interval_samples / sample_rate
            
            # Criterio 1: Proximidad Temporal
            if interval_samples < dt_threshold_samples:
                
                # Criterio 2: Análisis del Valle (Breath Stacking)
                # Buscamos el punto mínimo entre los dos picos.
                # Si el valor mínimo es alto (lejos del 0 relativo), indica que no hubo exhalación.
                segment = norm_sig[idx_current:idx_next]
                valley_min = np.min(segment)
                
                # Umbral de valle: Si el valle está por encima del 20% de la amplitud, 
                # es probable que sea un doble disparo con stacking.
                stacking_severity = valley_min 
                
                event_data = {
                    "peak1": idx_current,
                    "peak2": idx_next,
                    "interval_sec": interval_seconds,
                    "stacking_idx": stacking_severity
                }
                dt_events.append(event_data)

    results["events"] = dt_events
    results["event_count"] = len(dt_events)
    results["detected"] = len(dt_events) > 0
    
    return results
    # ==========================================
# BLOQUE 3: Interfaz de Usuario y Flujo Principal
# ==========================================

def main():
    st.title("🩺 Detección de Asincronías: Fase 2")
    st.markdown("""
    Este módulo analiza formas de onda capturadas del ventilador para detectar **Doble Disparo**.
    Asegúrese de capturar una imagen clara donde la curva (Flujo o Presión) sea visible.
    """)
    
    # Sidebar de Configuración
    st.sidebar.header("Configuración del Algoritmo")
    sensibilidad = st.sidebar.slider("Sensibilidad de Detección", 0.0, 1.0, 0.5, help="Aumente para detectar picos más sutiles.")
    fs_estimada = st.sidebar.number_input("Frecuencia de Muestreo Estimada (px/s)", min_value=10, value=50, step=10)

    # Entrada de Cámara con Manejo de Errores
    img_buffer = st.camera_input("Capturar Pantalla del Ventilador")

    if img_buffer is not None:
        # 1. Leer la imagen desde el buffer
        bytes_data = img_buffer.getvalue()
        img_array = np.frombuffer(bytes_data, np.uint8)
        original_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if original_img is not None:
            st.image(original_img, caption="Imagen Capturada", channels="BGR", use_column_width=True)
            
            with st.spinner("Procesando imagen y extrayendo señal..."):
                # --- Extracción de Señal (Simulada/Simplificada para el ejemplo) ---
                # NOTA: En producción, aquí iría el pipeline completo de HSV -> Skeletonize.
                # Para este ejemplo funcional, convertimos la imagen a escala de grises y 
                # extraemos el perfil de intensidad de una línea central o usamos luminancia.
                
                # Método robusto simple: Convertir a grises, invertir (onda clara fondo oscuro)
                gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
                
                # Asumimos que la onda es más brillante o más oscura. 
                # Aquí tomamos la columna con el píxel más brillante como "y".
                # Este es un método heurístico rápido.
                signal_extracted = # Lista vacia corregida
                height, width = gray.shape
                
                # Recorremos el 80% central del ancho para evitar bordes
                start_col = int(width * 0.1)
                end_col = int(width * 0.9)
                
                for col in range(start_col, end_col):
                    column_data = gray[:, col]
                    # Encontrar la posición del valor máximo (brillo) o mínimo (tinta)
                    # Asumimos onda clara sobre fondo oscuro:
                    max_idx = np.argmax(column_data) 
                    # Invertimos coordenada Y para gráfico cartesiano (0 abajo)
                    y_val = height - max_idx 
                    signal_extracted.append(y_val)
                
                signal_np = np.array(signal_extracted)

            # --- Ejecución del Análisis Fase 2 ---
            analysis = analyze_double_trigger(signal_np, sample_rate=fs_estimada, sensitivity=sensibilidad)

            # --- Visualización de Resultados ---
            st.divider()
            st.subheader("Resultados del Análisis")

            # Métricas
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Ciclos Detectados", len(analysis["peaks"]))
            col2.metric("Eventos Doble Disparo", analysis["event_count"], 
                        delta="-Peligro" if analysis["detected"] else "Normal",
                        delta_color="inverse")
            
            # Gráfico Interactivo
            fig, ax = plt.subplots(figsize=(10, 4))
            # Plot Señal Suavizada
            ax.plot(analysis["signal_processed"], label='Forma de Onda', color='steelblue', linewidth=1.5)
            
            # Plot Picos
            peaks_x = analysis["peaks"]
            peaks_y = analysis["signal_processed"][peaks_x]
            ax.scatter(peaks_x, peaks_y, color='lime', s=50, label='Inspiración', zorder=5)
            
            # Plot Eventos DT
            if analysis["detected"]:
                for event in analysis["events"]:
                    p1 = event["peak1"]
                    p2 = event["peak2"]
                    y_h = analysis["signal_processed"][p1]
                    # Dibujar línea roja conectando el doble disparo
                    ax.plot([p1, p2], [y_h, analysis["signal_processed"][p2]], color='red', linewidth=3, linestyle='--')
                    ax.annotate('DT', xy=(p2, analysis["signal_processed"][p2]), xytext=(p2, y_h*1.2),
                                arrowprops=dict(facecolor='red', shrink=0.05))
            
            ax.set_title("Análisis Morfológico de Ventilación")
            ax.set_xlabel("Tiempo (muestras)")
            ax.set_ylabel("Amplitud (u.a.)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            
            if analysis["detected"]:
                st.warning("""
                ⚠️ **Alerta de Asincronía:** Se han detectado eventos compatibles con Doble Disparo. 
                Verifique si el Tiempo Inspiratorio del ventilador es menor al Tiempo Neural del paciente.
                Considere evaluar el nivel de sedación o ajustar el ciclado.
                """)
            else:
                st.success("Análisis completado: No se detectaron asincronías mayores en este segmento.")

        else:
            st.error("Error: No se pudo decodificar la imagen.")
    else:
        st.info("Esperando captura de imagen...")

if __name__ == "__main__":
    main()
