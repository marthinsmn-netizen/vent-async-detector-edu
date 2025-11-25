# ==========================================
# BLOQUE 1: Importaciones y Configuración
# ==========================================
import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter

# Configuración crítica para entornos sin display
import matplotlib
matplotlib.use('Agg')

# Configuración de página
st.set_page_config(
    page_title="Detector de Asincronías Ventilatorias",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# BLOQUE 2: Funciones Core de Procesamiento
# ==========================================
def analyze_double_trigger(signal_data, sample_rate=50, sensitivity=0.5):
    """
    Detecta eventos de Doble Disparo en una señal unidimensional de ventilación.
    signal_data: 1D array-like (list o np.array)
    sample_rate: muestras por segundo (px/s)
    sensitivity: float [0..1] (mayor = detecta picos más sutiles)
    Retorna un dict con keys: detected (bool), event_count (int), events (list),
    peaks (list de índices), signal_processed (np.array), message (str)
    """
    # Normalizar input
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

    # --- Paso 1: Preprocesamiento y suavizado ---
    try:
        # Ajustamos ventana para que sea impar y < longitud de señal
        window = 11
        if window >= signal.size:
            window = signal.size - 1 if (signal.size - 1) % 2 == 1 else signal.size - 2
            if window < 3:
                window = 3
        poly = 3
        smoothed = savgol_filter(signal, window_length=window, polyorder=min(poly, window-1))
    except Exception:
        smoothed = signal.copy()

    results["signal_processed"] = smoothed

    # --- Paso 2: Normalización ---
    sig_min, sig_max = np.min(smoothed), np.max(smoothed)
    if sig_max - sig_min == 0:
        results["message"] = "Señal plana o sin variación detectada."
        return results

    norm_sig = (smoothed - sig_min) / (sig_max - sig_min)

    # --- Paso 3: Configuración de parámetros para find_peaks ---
    prominence_val = max(0.01, 0.6 - (sensitivity * 0.5))  # margen mínimo menor para señales débiles
    min_dist_samples = max(1, int(0.2 * sample_rate))
    min_width_
