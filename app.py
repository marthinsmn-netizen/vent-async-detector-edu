import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
import matplotlib

# Usar backend no interactivo para evitar errores de hilos en Streamlit
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
    
    # 1. Detección de Picos
    prominencia = 0.15 
    distancia_min = int(0.15 * fs)
    picos, _ = find_peaks(signal, prominence=prominencia, distance=distancia_min)
    
    if len(picos) < 2:
        return hallazgos, picos

    # 2. Análisis de Pares de Picos
    for i in range(len(picos) - 1):
        p1 = picos[i]
        p2 = picos[i+1]
        distancia_tiempo = (p2 - p1) / fs
        
        # Si dos picos están a menos de 1.0 segundos, hay algo raro
        if distancia_tiempo < 1.0:
            segmento = signal[p1:p2]
            valle_idx = np.argmin(segmento)
            altura_valle = segmento[valle_idx]
            altura_pico1 = signal[p1]
            
            # Evitar división por cero
            if altura_pico1 == 0: altura_pico1 = 0.001
            ratio_valle = altura_valle / altura_pico1
            
            # --- Lógica de Decisión ---
            if tipo_curva == "Presión":
                if ratio_valle > 0.6: 
                    hallazgos["diagnostico"] = "Hambre de Flujo (Flow Starvation)"
                    hallazgos["color"] = "orange"
                    hallazgos["explicacion"] = "Muesca cóncava detectada en la rama inspiratoria."
                    hallazgos["consejo"] = "El paciente demanda más aire del entregado.\n\n👉 **Acción:** Aumente el Flujo Inspiratorio o reduzca el Rise Time."
                    return hallazgos, picos
                
                elif ratio_valle < 0.5:
                    hallazgos["diagnostico"] = "Doble Disparo (Double Trigger)"
                    hallazgos["color"] = "red"
                    hallazgos["explicacion"] = "Dos ciclos consecutivos detectados debido a un Tiempo Inspiratorio neural prolongado."
                    hallazgos["consejo"] = "👉 **Acción:** Aumente el Tiempo Inspiratorio (Ti) o el Volumen Tidal."
                    return hallazgos, picos
            
            elif tipo_curva == "Flujo":
                if ratio_valle < 0.3:
                    hallazgos["diagnostico"] = "Posible Doble Disparo"
                    hallazgos["color"] = "red"
                    hallazgos["explicacion"] = "Reinicio del flujo inspiratorio antes de exhalación completa."
                    hallazgos["consejo"] = "Evalúe nivel de sedación o ajuste el Ti mecánico."
                    return hallazgos, picos

    return hallazgos, picos

# ==========================================
# INTERFAZ DE USUARIO (GUI)
# ==========================================

def main():
    st.title("🫁 Ventilator Lab: Guía Clínica")
    st.markdown("Herramienta educativa para la detección de asincronías.")

    # 1. Selector de Contexto
    tipo = st.radio("¿Qué curva estás analizando?", 
                   ["Presión (Paw)", "Flujo (Flow)"], 
                   horizontal=True)
    
    # 2. Cámara
    imagen = st.camera_input("Toma una foto a la pantalla del ventilador")

    if imagen:
        # --- PROCESAMIENTO DE IMAGEN CON CALIBRACIÓN ---
        bytes_data = imagen.getvalue()
        img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        
        # Conversión a HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, w, _ = img.shape

        # --- SECCIÓN DE CALIBRACIÓN (SLIDERS) ---
        st.sidebar.header("⚙️ Calibración de Color")
        st.sidebar.info("Ajusta estos valores si la curva no se detecta correctamente.")

        # Valores por defecto según el tipo de curva
        if "Presión" in tipo:
            # Amarillo
            def_h, def_s, def_v = (20, 40), (100, 255), (100, 255)
        else:
            # Cian/Azul
            def_h, def_s, def_v = (80, 100), (100, 255), (100, 255)

        # Sliders únicos por tipo de curva (usando key dinámica)
        h_min, h_max = st.sidebar.slider(f"Matiz (H) - {tipo}", 0, 179, def_h)
        s_min, s_max = st.sidebar.slider(f"Saturación (S) - {tipo}", 0, 255, def_s)
        v_min, v_max = st.sidebar.slider(f"Brillo (V) - {tipo}", 0, 255, def_v)

        lower_color = np.array([h_min, s_min, v_min])
        upper_color = np.array([h_max, s_max, v_max])

        # Crear máscara
        mask = cv2.inRange(hsv, lower_color, upper_color)

        # Mostrar "Visión de la IA" para debug
        with st.expander("👁️ Ver lo que detecta la IA (Calibración)", expanded=False):
            st.image(mask, caption="Máscara de Color (Blanco = Detectado)", use_column_width=True)
            st.caption("Si ves mucho ruido blanco o la pantalla negra, ajusta los sliders en la izquierda.")

        # --- EXTRACCIÓN DE SEÑAL ---
        raw_signal = []
        for col in range(int(w*0.1), int(w*0.9)): # Recorte márgenes 10%
            col_data = mask[:, col]
            
            if np.max(col_data) > 0:
                # Detectar el píxel blanco más alto (coordenada Y invertida)
                y_pos = h - np.argmax(col_data)
                raw_signal.append(y_pos)
            else:
                # Si no hay señal, mantener el valor anterior
                val = raw_signal[-1] if len(raw_signal) > 0 else 0
                raw_signal.append(val)
        
        # Validación de señal vacía
        if np.max(raw_signal) == 0:
            st.error("⚠️ No se ha detectado ninguna curva. Abre la barra lateral (izquierda) y ajusta los colores.")
            st.stop()

        # Normalización y Suavizado
        sig_np = np.array(raw_signal)
        sig_norm = (sig_np - np.min(sig_np)) / (np.max(sig_np) - np.min(sig_np) + 1e-6)
        
        try:
            # Suavizado para reducir el ruido de la máscara binaria
            sig_smooth = savgol_filter(sig_norm, 31, 3)
        except:
            sig_smooth = sig_norm

        # 3. Análisis Clínico
        # Pasamos solo la primera palabra ("Presión" o "Flujo")
        resultado, picos = analizar_curva(sig_smooth, tipo.split()[0])

        # 4. Visualización de Resultados
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

        # Gráfico
        fig, ax = plt.subplots(figsize=(10, 3))
        fig.patch.set_facecolor('#0e1117')
        ax.set_facecolor('black')
        
        color_linea = 'yellow' if "Presión" in tipo else 'cyan'
        ax.plot(sig_smooth, color=color_linea, lw=2, label="Señal Detectada")
        ax.plot(picos, sig_smooth[picos], "wo", markersize=5) # Picos marcados
        
        ax.axis('off')
        st.pyplot(fig)

if __name__ == "__main__":
    main()
