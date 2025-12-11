import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
import matplotlib
matplotlib.use('Agg')

# --- Configuración Estética ---
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
    hallazgos = {
        "diagnostico": "Normal",
        "color": "green",
        "explicacion": "Trazo estable. Sincronía aceptable.",
        "consejo": "Continúe monitorizando la mecánica pulmonar."
    }
    
    # Detección de Picos
    prominencia = 0.15 
    distancia_min = int(0.15 * fs)
    picos, propiedades = find_peaks(signal, prominence=prominencia, distance=distancia_min)
    
    if len(picos) < 2:
        return hallazgos, picos

    # Análisis de Pares
    # CORRECCIÓN 1: Eliminé la variable 'eventos_detectados' que no se usaba
    
    for i in range(len(picos) - 1):
        p1 = picos[i]
        p2 = picos[i+1]
        distancia_tiempo = (p2 - p1) / fs
        
        if distancia_tiempo < 1.0:
            segmento = signal[p1:p2]
            valle_idx = np.argmin(segmento)
            altura_valle = segmento[valle_idx]
            altura_pico1 = signal[p1]
            
            # Evitar división por cero
            if altura_pico1 == 0: altura_pico1 = 0.001
            ratio_valle = altura_valle / altura_pico1
            
            # --- Lógica de Decisión ---
            # El string debe coincidir exactamente con lo que enviamos desde main
            if tipo_curva == "Presión":
                if ratio_valle > 0.6: 
                    hallazgos["diagnostico"] = "Hambre de Flujo (Flow Starvation)"
                    hallazgos["color"] = "orange"
                    hallazgos["explicacion"] = "Muesca cóncava en la rama inspiratoria."
                    hallazgos["consejo"] = "👉 **Acción:** Aumente el Flujo Inspiratorio o reduzca el Rise Time."
                    return hallazgos, picos
                
                elif ratio_valle < 0.5:
                    hallazgos["diagnostico"] = "Doble Disparo (Double Trigger)"
                    hallazgos["color"] = "red"
                    hallazgos["explicacion"] = "Dos ciclos consecutivos por tiempo neural prolongado."
                    hallazgos["consejo"] = "👉 **Acción:** Aumente el Tiempo Inspiratorio o el Volumen Tidal."
                    return hallazgos, picos
            
            elif tipo_curva == "Flujo":
                if ratio_valle < 0.3:
                    hallazgos["diagnostico"] = "Posible Doble Disparo"
                    hallazgos["color"] = "red"
                    hallazgos["explicacion"] = "Reinicio de flujo antes de exhalación."
                    hallazgos["consejo"] = "Evalúe sedación o ajuste Ti."
                    return hallazgos, picos

    return hallazgos, picos

# ==========================================
# INTERFAZ DE USUARIO (GUI)
# ==========================================

def main():
    st.title("🫁 Ventilator Lab: Guía Clínica")
    st.markdown("Herramienta educativa para la detección de asincronías.")

    tipo = st.radio("¿Qué curva estás analizando?", 
                   ["Presión (Paw)", "Flujo (Flow)"], 
                   horizontal=True)
    
    imagen = st.camera_input("Toma una foto a la pantalla del ventilador")

    if imagen:
        bytes_data = imagen.getvalue()
        img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        h, w = gray.shape
        # CORRECCIÓN 2: Inicialización de lista
        raw_signal = [] 
        
        for col in range(int(w*0.1), int(w*0.9)):
            col_data = gray[:, col]
            raw_signal.append(h - np.argmax(col_data))
        
        sig_np = np.array(raw_signal)
        sig_norm = (sig_np - np.min(sig_np)) / (np.max(sig_np) - np.min(sig_np) + 1e-6)
        
        try:
            sig_smooth = savgol_filter(sig_norm, 15, 3)
        except:
            sig_smooth = sig_norm

        # CORRECCIÓN 3: Pasar solo la primera palabra ("Presión" o "Flujo")
        resultado, picos = analizar_curva(sig_smooth, tipo.split()[0])

        st.divider()
        
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

        with st.expander("🎓 ¿Qué debo hacer? (Guía Clínica)", expanded=True):
            st.markdown(resultado["consejo"])

        fig, ax = plt.subplots(figsize=(10, 3))
        fig.patch.set_facecolor('#0e1117')
        ax.set_facecolor('black')
        
        color_linea = 'yellow' if "Presión" in tipo else 'cyan'
        ax.plot(sig_smooth, color=color_linea, lw=2)
        ax.plot(picos, sig_smooth[picos], "wo", markersize=5)
        
        ax.axis('off')
        st.pyplot(fig)

if __name__ == "__main__":
    main()
