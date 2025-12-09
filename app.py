import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
import matplotlib
matplotlib.use('Agg')

# --- Configuración de la App ---
st.set_page_config(
    page_title="Ventilador Lab - Asistente Educativo",
    page_icon="🫁",
    layout="centered" # Diseño más tipo "app móvil"
)

# ==========================================
# LÓGICA CLÍNICA (EL CEREBRO)
# ==========================================

def analizar_curva_presion(signal, fs=50):
    """
    Analiza la forma de la curva de PRESIÓN para distinguir
    entre Hambre de Flujo (mordida) y Doble Disparo (dos picos).
    """
    hallazgos = {
        "tipo": "Normal",
        "confianza": 0.0,
        "mensaje": "Patrón ventilatorio aceptable.",
        "accion": "Continuar monitorización."
    }
    
    # 1. Detectar Picos Principales (Inspiraciones)
    picos, _ = find_peaks(signal, prominence=0.2, distance=int(0.5*fs))
    
    if len(picos) < 2:
        hallazgos["mensaje"] = "No se detectan suficientes ciclos para un diagnóstico."
        return hallazgos, picos

    # 2. Buscar Doble Disparo (Criterio de Tiempo)
    # Si dos picos están muy cerca (< 0.8 seg), es Doble Disparo
    for i in range(len(picos)-1):
        tiempo_entre_picos = (picos[i+1] - picos[i]) / fs
        if tiempo_entre_picos < 0.8: # Menos de 0.8s entre respiraciones
            hallazgos["tipo"] = "Doble Disparo"
            hallazgos["mensaje"] = f"Se detectaron dos ciclos en {tiempo_entre_picos:.2f} segundos."
            hallazgos["accion"] = "Posible tiempo inspiratorio corto. El paciente quiere más aire o tiempo."
            return hallazgos, picos

    # 3. Buscar Hambre de Flujo (Criterio de Forma)
    # Si no es doble disparo, miramos si la subida tiene "panza" (concavidad)
    # Analizamos el primer ciclo completo
    inicio = max(0, picos - int(0.4*fs)) # Asumimos inicio 0.4s antes del pico
    fin = picos
    segmento = signal[inicio:fin]
    
    if len(segmento) > 5:
        # Creamos una línea ideal recta
        linea_ideal = np.linspace(segmento, segmento[-1], len(segmento))
        # Calculamos cuánto se aleja la curva real hacia abajo
        diferencia = linea_ideal - segmento
        max_depresion = np.max(diferencia)
        altura_pico = np.max(signal) - np.min(signal)
        
        # Si la "panza" es > 15% de la altura, es Hambre de Flujo
        if (max_depresion / altura_pico) > 0.15:
            hallazgos["tipo"] = "Hambre de Flujo"
            hallazgos["mensaje"] = "La curva de presión se hunde durante la entrada de aire."
            hallazgos["accion"] = "El flujo es insuficiente para la demanda del paciente."
            return hallazgos, picos

    return hallazgos, picos

# ==========================================
# INTERFAZ PARA EL ALUMNO
# ==========================================

def main():
    st.title("🫁 Asistente de Asincronías")
    st.write("Toma una foto a la pantalla del ventilador para recibir orientación.")

    # Selector simple
    modo = st.radio("¿Qué curva estás viendo?", ["Presión (Paw)", "Flujo (Flow)"], horizontal=True)

    # Cámara
    img_file = st.camera_input("Capturar Pantalla")

    if img_file is not None:
        # 1. Procesamiento de Imagen (Digitalización)
        bytes_data = img_file.getvalue()
        img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Extraer señal (perfil de brillo)
        h, w = gray.shape
        signal =
        # Analizamos el centro de la imagen para evitar bordes
        for col in range(int(w*0.1), int(w*0.9)):
            col_data = gray[:, col]
            y_val = h - np.argmax(col_data) # Invertir eje Y
            signal.append(y_val)
        
        # Normalizar señal (0 a 1)
        signal = np.array(signal)
        signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal) + 1e-6)

        # 2. Diagnóstico Automático
        if "Presión" in modo:
            resultado, picos = analizar_curva_presion(signal)
        else:
            # Lógica simplificada para Flujo (solo detección de picos por ahora)
            picos, _ = find_peaks(signal, prominence=0.3)
            resultado = {"tipo": "Análisis de Flujo", "mensaje": "Visualizando ciclos.", "accion": "Verifique retorno a cero."}

        # 3. Mostrar Resultados Claros
        st.divider()
        
        # Semáforo de Diagnóstico
        if resultado["tipo"] == "Normal":
            st.success(f"✅ **Diagnóstico:** {resultado['tipo']}")
        else:
            st.error(f"⚠️ **Diagnóstico:** {resultado['tipo']}")
        
        st.info(f"ℹ️ **Interpretación:** {resultado['mensaje']}")

        # 4. Guía de Toma de Decisiones (Educativo)
        with st.expander("🎓 Guía Clínica: ¿Qué debo hacer?", expanded=True):
            if resultado["tipo"] == "Doble Disparo":
                st.markdown("""
                **El paciente está realizando dos esfuerzos seguidos.**
                *   **Causa:** El tiempo inspiratorio programado es muy corto para el paciente (Ti Neural > Ti Mecánico).
                *   **Acción Sugerida:** 
                    1. Aumentar el **Tiempo Inspiratorio** o el Volumen Tidal.
                    2. Si es por dolor/ansiedad, evaluar analgesia.
                """)
            elif resultado["tipo"] == "Hambre de Flujo":
                st.markdown("""
                **El paciente está 'chupando' aire con fuerza.**
                *   **Causa:** El flujo de entrega es muy lento o bajo para la demanda del paciente.
                *   **Acción Sugerida:**
                    1. Aumentar la **Velocidad de Flujo** (L/min).
                    2. Cambiar el "Rise Time" (tiempo de subida) para que sea más rápido.
                    3. Considerar cambio a Modalidad de Presión Soporte.
                """)
            else:
                st.markdown("""
                **Patrón estable.**
                *   Continúe vigilando la sincronía.
                *   Verifique que no haya fugas (la curva debe volver a cero).
                """)

        # 5. Gráfico de Referencia (Feedback visual)
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(signal, color='yellow' if "Presión" in modo else 'cyan', lw=2)
        ax.plot(picos, signal[picos], "x", color='red')
        ax.set_facecolor('#000000') # Fondo negro tipo ventilador
        fig.patch.set_facecolor('#0e1117') # Fondo oscuro de Streamlit
        ax.axis('off') # Quitar ejes feos
        st.pyplot(fig)

if __name__ == "__main__":
    main()
