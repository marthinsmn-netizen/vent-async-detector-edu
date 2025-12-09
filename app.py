import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
import matplotlib
matplotlib.use('Agg')

# --- Configuración Estética (Modo App Móvil) ---
st.set_page_config(
    page_title="Asistente Ventilación",
    page_icon="🫁",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ==========================================
# LÓGICA DE INTELIGENCIA CLÍNICA
# ==========================================

def analizar_curva(signal, tipo_curva, fs=50):
    """
    Analiza la señal usando reglas heurísticas (Valle Profundo vs Superficial)
    para distinguir entre Doble Disparo y Hambre de Flujo.
    """
    hallazgos = {
        "diagnostico": "Normal",
        "color": "green",
        "explicacion": "Trazo estable. Sincronía aceptable.",
        "consejo": "Continúe monitorizando la mecánica pulmonar."
    }
    
    # 1. Detección de Picos (Candidatos a respiraciones)
    # Usamos parámetros generosos para capturar todo, luego filtramos
    prominencia = 0.15 # Sensibilidad media
    distancia_min = int(0.15 * fs) # 150ms
    picos, propiedades = find_peaks(signal, prominence=prominencia, distance=distancia_min)
    
    if len(picos) < 2:
        return hallazgos, picos

    # 2. Análisis de Pares de Picos (Busca eventos cercanos)
    eventos_detectados =
    
    for i in range(len(picos) - 1):
        p1 = picos[i]
        p2 = picos[i+1]
        distancia_tiempo = (p2 - p1) / fs
        
        # Si dos picos están a menos de 1.0 segundos, hay algo raro
        if distancia_tiempo < 1.0:
            # --- LA REGLA DEL VALLE (El Discriminador) ---
            # Buscamos el punto más bajo entre los dos picos
            segmento = signal[p1:p2]
            valle_idx = np.argmin(segmento)
            altura_valle = segmento[valle_idx]
            altura_pico1 = signal[p1]
            
            # Calculamos qué tanto bajó la señal (Ratio de Caída)
            # 0.0 = Bajó hasta el suelo (Exhalación completa)
            # 1.0 = No bajó nada (Línea recta)
            ratio_valle = altura_valle / altura_pico1
            
            # --- Lógica de Decisión ---
            if tipo_curva == "Presión":
                if ratio_valle > 0.6: 
                    # El valle es ALTO (bajó poco). Es una sola respiración deformada (muesca).
                    hallazgos["diagnostico"] = "Hambre de Flujo (Flow Starvation)"
                    hallazgos["color"] = "orange"
                    hallazgos["explicacion"] = "La curva de presión tiene una concavidad ('muesca') durante la subida."
                    hallazgos["consejo"] = "El paciente 'chupa' aire más rápido de lo que el ventilador entrega.\n\n👉 **Acción:** Aumente el Flujo Inspiratorio o reduzca el Rise Time."
                    return hallazgos, picos
                
                elif ratio_valle < 0.5:
                    # El valle es BAJO (bajó mucho). Son dos intentos separados.
                    hallazgos["diagnostico"] = "Doble Disparo (Double Trigger)"
                    hallazgos["color"] = "red"
                    hallazgos["explicacion"] = "Se detectan dos ciclos muy seguidos con exhalación incompleta."
                    hallazgos["consejo"] = "El Tiempo Inspiratorio (Ti) neural del paciente es más largo que el programado.\n\n👉 **Acción:** Aumente el Tiempo Inspiratorio o el Volumen Tidal."
                    return hallazgos, picos
            
            elif tipo_curva == "Flujo":
                # En flujo, los picos cercanos suelen ser Doble Disparo o Autociclado
                if ratio_valle < 0.3: # Bajó casi a cero
                    hallazgos["diagnostico"] = "Posible Doble Disparo"
                    hallazgos["color"] = "red"
                    hallazgos["explicacion"] = "Reinicio del flujo inspiratorio antes de exhalación completa."
                    hallazgos["consejo"] = "Evalúe sedación o ajuste el Ti mecánico."
                    return hallazgos, picos

    # Si llegamos aquí, revisamos Esfuerzos Inefectivos (solo en Flujo)
    if tipo_curva == "Flujo":
        # Buscamos picos pequeños en la zona negativa/baja (exhalación)
        # Simplificación para esta demo
        pass

    return hallazgos, picos

# ==========================================
# INTERFAZ DE USUARIO (GUI)
# ==========================================

def main():
    st.title("🫁 Ventilator Lab: Guía Clínica")
    st.markdown("Herramienta educativa para la detección de asincronías.")

    # 1. Selector de Contexto (Simple)
    tipo = st.radio("¿Qué curva estás analizando?", 
                   ["Presión (Paw)", "Flujo (Flow)"], 
                   horizontal=True)
    
    # 2. Cámara
    imagen = st.camera_input("Toma una foto a la pantalla del ventilador")

    if imagen:
        # Procesamiento de imagen (Fase 1 simplificada)
        bytes_data = imagen.getvalue()
        img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Extracción de señal (Perfil de brillo inverso)
        h, w = gray.shape
        raw_signal =
        for col in range(int(w*0.1), int(w*0.9)): # Recorte márgenes
            col_data = gray[:, col]
            # Asumimos curva clara sobre fondo oscuro (argmax)
            # Invertimos Y para que sea intuitivo (0 abajo)
            raw_signal.append(h - np.argmax(col_data))
        
        # Normalización (0.0 a 1.0) para que la "Regla del Valle" funcione igual en todos los celulares
        sig_np = np.array(raw_signal)
        sig_norm = (sig_np - np.min(sig_np)) / (np.max(sig_np) - np.min(sig_np) + 1e-6)
        
        # Suavizado suave para quitar ruido de la cámara
        try:
            sig_smooth = savgol_filter(sig_norm, 15, 3)
        except:
            sig_smooth = sig_norm

        # 3. Análisis
        resultado, picos = analizar_curva(sig_smooth, tipo.split())

        # 4. Resultados Visuales
        st.divider()
        
        # Tarjeta de Diagnóstico
        col_a, col_b = st.columns([1, 2])
        with col_a:
            if resultado["color"] == "green":
                st.success(f"✅ {resultado['diagnostico']}")
            elif resultado["color"] == "orange":
                st.warning(f"⚠️ {resultado['diagnostico']}")
            else:
                st.error(f"🚨 {resultado['diagnostico']}")
        
        with col_b:
            st.info(f"**Interpretación:** {resultado['explicacion']}")

        # Guía de Acción
        with st.expander("🎓 ¿Qué debo hacer? (Guía Clínica)", expanded=True):
            st.markdown(resultado["consejo"])

        # Gráfico de Validación
        fig, ax = plt.subplots(figsize=(10, 3))
        # Fondo oscuro médico
        fig.patch.set_facecolor('#0e1117')
        ax.set_facecolor('black')
        
        color_linea = 'yellow' if "Presión" in tipo else 'cyan'
        ax.plot(sig_smooth, color=color_linea, lw=2)
        ax.plot(picos, sig_smooth[picos], "wo", markersize=5) # Picos en blanco
        
        ax.axis('off') # Limpio, sin ejes
        st.pyplot(fig)

if __name__ == "__main__":
    main()
