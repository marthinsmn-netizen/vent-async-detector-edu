import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import matplotlib
matplotlib.use("Agg")

# ------------------------------------------------------------
# CONFIGURACIÓN DE LA APP
# ------------------------------------------------------------
st.set_page_config(
    page_title="Ventilador Lab - Asistente Educativo",
    page_icon="🫁",
    layout="centered"
)

# ------------------------------------------------------------
# CSS INTELIGENTE: adapta colores según modo claro/oscuro
# ------------------------------------------------------------
st.markdown("""
<style>

/* ======= MODO OSCURO ======= */
@media (prefers-color-scheme: dark) {
    html, body, [class*="css"] {
        color: #F2F2F2 !important;
        background-color: #0E1117 !important;
    }
    h1, h2, h3, h4, h5, h6, label, p, span, div {
        color: #FFFFFF !important;
    }
    .stRadio label, .stSelectbox label {
        color: #FFFFFF !important;
    }
}

/* ======= MODO CLARO ======= */
@media (prefers-color-scheme: light) {
    html, body, [class*="css"] {
        color: #202020 !important;
        background-color: #FAFAFA !important;
    }
    h1, h2, h3, h4, h5, h6, label, p, span, div {
        color: #000000 !important;
    }
    .stRadio label, .stSelectbox label {
        color: #000000 !important;
    }
}

/* ======= FIJAR TEXTOS EN WIDGETS ======= */
.stMarkdown, .stText, .stRadio, .stSelectbox, .stCameraInput {
    color: inherit !important;
}

</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# PANEL LATERAL (Explicaciones)
# ------------------------------------------------------------
with st.sidebar:
    st.header("ℹ️ Parámetros del Análisis")

    st.subheader("SG Windows Odd")
    st.write("""
    Es el número de muestras usadas para suavizar la señal.
    Debe ser impar. Ventanas mayores = señal más suave, pero menos detalles.
    """)

    st.subheader("SG Polyorder")
    st.write("""
    Es el grado del polinomio ajustado dentro de cada ventana.
    Valores altos suavizan menos y preservan más la forma original.
    """)

    st.subheader("Prominence (picos principales)")
    st.write("""
    Es qué tan "alto" debe ser un pico respecto a su entorno.
    A mayor prominence, se detectan menos picos pero más confiables.
    """)

    st.subheader("IE prominence scale")
    st.write("""
    Normaliza la prominencia respecto a la amplitud del ciclo.
    Útil para comparar picos entre curvas con diferentes intensidades.
    """)

# ------------------------------------------------------------
# LÓGICA CLÍNICA
# ------------------------------------------------------------
def analizar_curva_presion(signal, fs=50):
    hallazgos = {
        "tipo": "Normal",
        "mensaje": "Patrón ventilatorio aceptable.",
        "accion": "Continuar monitorización."
    }

    picos, _ = find_peaks(signal, prominence=0.2, distance=int(0.5 * fs))

    if len(picos) < 2:
        hallazgos["mensaje"] = "Se detectaron pocos ciclos para diagnóstico confiable."
        return hallazgos, picos

    # Detectar doble disparo (< 0.8 segundos entre picos)
    for i in range(len(picos) - 1):
        t = (picos[i+1] - picos[i]) / fs
        if t < 0.8:
            hallazgos["tipo"] = "Doble Disparo"
            hallazgos["mensaje"] = f"Dos esfuerzos detectados en {t:.2f} s."
            hallazgos["accion"] = "Aumentar el tiempo inspiratorio o ajustar soporte."
            return hallazgos, picos

    # Buscar "panza" en la curva (hambre de flujo)
    idx = picos[0]
    inicio = max(0, idx - int(0.4 * fs))
    segmento = signal[inicio:idx]

    if len(segmento) > 5:
        linea = np.linspace(segmento[0], segmento[-1], len(segmento))
        diferencia = linea - segmento
        max_dep = np.max(diferencia)
        altura = np.max(signal) - np.min(signal)

        if altura > 0 and (max_dep / altura) > 0.15:
            hallazgos["tipo"] = "Hambre de Flujo"
            hallazgos["mensaje"] = "Concavidad inspiratoria marcada."
            hallazgos["accion"] = "Aumentar flujo o modificar rise time."
            return hallazgos, picos

    return hallazgos, picos

# ------------------------------------------------------------
# INTERFAZ PRINCIPAL
# ------------------------------------------------------------
def main():
    st.title("🫁 Asistente de Asincronías")
    st.write("Subí o toma una foto de la pantalla del ventilador.")

    modo = st.radio("Selecciona curva analizada", ["Presión (Paw)", "Flujo (Flow)"], horizontal=True)

    img_file = st.camera_input("Capturar Imagen del Ventilador")

    if img_file is not None:

        # Leer imagen
        bytes_data = img_file.getvalue()
        img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Extraer señal vertical
        h, w = gray.shape
        signal = []
        for col in range(int(w * 0.1), int(w * 0.9)):
            col_data = gray[:, col]
            y = h - np.argmax(col_data)
            signal.append(y)

        signal = np.array(signal)
        signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal) + 1e-6)

        # Diagnóstico
        if "Presión" in modo:
            resultado, picos = analizar_curva_presion(signal)
        else:
            picos, _ = find_peaks(signal, prominence=0.3)
            resultado = {
                "tipo": "Flujo",
                "mensaje": "Ciclos detectados visualmente.",
                "accion": "Verificar retorno a cero y forma del flujo."
            }

        # Mostrar resultados
        st.divider()

        if resultado["tipo"] == "Normal":
            st.success(f"✅ Diagnóstico: {resultado['tipo']}")
        else:
            st.error(f"⚠️ Diagnóstico: {resultado['tipo']}")

        st.info(f"ℹ️ {resultado['mensaje']}")

        # GUIA CLÍNICA
        with st.expander("🎓 Explicación Clínica"):
            if resultado["tipo"] == "Doble Disparo":
                st.markdown("""
                **Doble Disparo:** El paciente realiza dos esfuerzos seguidos.  
                - Tiempo inspiratorio mecánico demasiado corto.  
                - Ajustar **Ti**, analgesia o confort.
                """)
            elif resultado["tipo"] == "Hambre de Flujo":
                st.markdown("""
                **Hambre de Flujo:** La presión cae durante la inspiración.  
                Indica que el ventilador entrega **menos flujo del que el paciente demanda**.  
                - Aumentar flujo  
                - Ajustar rise time  
                - Considerar PS
                """)
            else:
                st.markdown("Patrón estable. Mantener vigilancia.")

        # Gráfica
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(signal, color='yellow' if "Presión" in modo else 'cyan', lw=2)
        ax.plot(picos, signal[picos], "x", color='red')
        ax.set_facecolor('#000')
        ax.axis('off')
        st.pyplot(fig)

# ------------------------------------------------------------
if __name__ == "__main__":
    main()
