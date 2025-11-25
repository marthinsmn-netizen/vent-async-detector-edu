# app.py
import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
import matplotlib
import time
import io
import json

# Usar backend no interactivo para evitar errores de hilos en servidor
matplotlib.use('Agg')

# --- CONFIGURACIÓN DE LA APP (UX PROFESIONAL) ---
st.set_page_config(
    page_title="Ventilator Lab - Asistente Avanzado",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# ESTILOS CSS PROFESIONALES (Whitelabeling)
# ==========================================
def local_css():
    estilo_medico = """
        <style>
        /* 1. OCULTAR ELEMENTOS DE STREAMLIT */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* 2. FONDO Y FUENTES (Look & Feel Clínico) */
        .stApp {
            background-color: #F4F6F9;
            font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        }

        /* 3. PERSONALIZACIÓN DE TÍTULOS */
        h1 {
            color: #1E3A8A;
            font-weight: 800;
            padding-bottom: 10px;
            border-bottom: 2px solid #3B82F6;
        }
        h2, h3 {
            color: #2C3E50;
            font-weight: 600;
        }

        /* 4. BOTONES MODERNOS */
        div.stButton > button {
            background-color: #0284C7;
            color: white;
            border-radius: 8px;
            border: none;
            padding: 0.6rem 1.2rem;
            font-weight: 600;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
        }
        div.stButton > button:hover {
            background-color: #0369A1;
            transform: translateY(-2px);
            box-shadow: 0 4px 6px rgba(0,0,0,0.15);
        }

        /* 5. TARJETAS DE MÉTRICAS */
        [data-testid="stMetric"] {
            background-color: #FFFFFF;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            border-left: 5px solid #0284C7;
        }

        /* 6. PANELES Y GRÁFICOS */
        .stPlotlyChart, div[data-testid="stExpander"] {
            background-color: #FFFFFF;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        }
        
        /* 7. ALERTA PERSONALIZADA */
        .stSuccess, .stInfo, .stWarning, .stError {
            border-radius: 8px;
            border-left: 5px solid rgba(0,0,0,0.1);
        }
        </style>
    """
    return estilo_medico

# =========================
# UTIL / DIGITALIZACIÓN
# =========================

def extract_signal_from_image(gray, col_start_ratio=0.05, col_end_ratio=0.95, invert=True):
    """
    Extrae una señal 1D desde una imagen en escala de grises usando centroid por columna.
    """
    if gray is None:
        return np.array([], dtype=float)
    h, w = gray.shape
    start_col = int(w * col_start_ratio)
    end_col = int(w * col_end_ratio)
    if end_col <= start_col:
        start_col = 0
        end_col = w
    cols = range(start_col, end_col)

    signal = []
    for col in cols:
        col_data = gray[:, col].astype(float)
        if invert:
            weights = col_data
        else:
            weights = 255.0 - col_data

        s = np.sum(weights)
        if s <= 1e-6:
            idx = int(np.argmax(col_data))
            y = float(h - idx)
        else:
            indices = np.arange(h, dtype=float)
            centroid = float(np.sum(indices * weights) / s)
            y = float(h - centroid)
        signal.append(y)

    return np.array(signal, dtype=float)

def robust_smooth(signal, window=11, poly=3):
    """Savitzky-Golay with safe window sizing, plus light gaussian blur."""
    sig = np.asarray(signal, dtype=float)
    n = sig.size
    if n < 5:
        return sig.copy()
    win = int(window)
    if win >= n:
        win = n - 1 if (n - 1) % 2 == 1 else n - 2
    win = max(3, win if win % 2 == 1 else win - 1)
    try:
        sg = savgol_filter(sig, window_length=win, polyorder=min(poly, win-1))
    except Exception:
        sg = sig.copy()
    kern = np.exp(-0.5 * (np.linspace(-1, 1, 5) ** 2) / (0.2 ** 2))
    kern = kern / np.sum(kern)
    sg2 = np.convolve(sg, kern, mode='same')
    return sg2

# =========================
# ONSET / OFFSET DETECTION
# =========================

def detect_onset(signal, peak_idx, fs, back_search_sec=0.5, slope_thresh_rel=0.2):
    n = len(signal)
    back_samples = max(1, int(back_search_sec * fs))
    start = max(0, int(peak_idx) - back_samples)
    seg = signal[start:int(peak_idx)+1]
    if seg.size < 3:
        return start
    deriv = np.gradient(seg)
    max_slope = np.max(np.abs(deriv)) + 1e-9
    thr = slope_thresh_rel * max_slope
    for i in range(len(deriv)-1, -1, -1):
        if deriv[i] <= thr:
            onset = start + min(i+1, len(seg)-1)
            return int(onset)
    return int(start)

def detect_offset(signal, peak_idx, fs, forward_search_sec=1.0):
    n = len(signal)
    fwd_samples = max(1, int(forward_search_sec * fs))
    end = min(n-1, int(peak_idx) + fwd_samples)
    seg = signal[int(peak_idx):end+1]
    if seg.size < 3:
        return end
    deriv = np.gradient(seg)
    for i in range(1, len(deriv)):
        if deriv[i] < 0:
            return int(int(peak_idx) + i)
    return int(end)

# =========================
# EVENT DETECTION (B, C, D)
# =========================

def detect_ineffective_efforts(signal, major_peaks, fs, ie_prom_scale=0.06):
    """Detecta esfuerzos inefectivos: micro picos en la fase espiratoria."""
    ie_events = []
    sig = np.asarray(signal, dtype=float)
    if len(major_peaks) < 2 or sig.size == 0:
        return ie_events

    peak_amps = sig[np.array(major_peaks)]
    median_amp = float(np.median(peak_amps)) if peak_amps.size > 0 else 1.0
    prominence = max(1e-3, ie_prom_scale * median_amp)

    for i in range(len(major_peaks)-1):
        s = int(major_peaks[i])
        e = int(major_peaks[i+1])
        if e - s < 5:
            continue
        s_zone = s + int((e-s) * 0.3)
        e_zone = s + int((e-s) * 0.9)
        if e_zone <= s_zone:
            continue
        segment = sig[s_zone:e_zone]
        micro_peaks, _ = find_peaks(segment, prominence=prominence, distance=max(1, int(0.05*fs)))
        for mp in micro_peaks:
            global_idx = s_zone + int(mp)
            if not any(abs(global_idx - p) < max(1, int(0.2*fs)) for p in major_peaks):
                ie_events.append(int(global_idx))
    return sorted(list(set(ie_events)))

def detect_auto_trigger(signal, major_peaks, fs):
    """Detecta auto-triggering: intervalos cortos y amplitudes pequeñas."""
    at_events = []
    sig = np.asarray(signal, dtype=float)
    if len(major_peaks) < 4:
        return at_events
    intervals = np.diff(np.array(major_peaks))
    median_int = float(np.median(intervals)) if intervals.size>0 else 0.0
    median_amp = float(np.median(sig[np.array(major_peaks)])) if len(major_peaks)>0 else 0.0
    for i, d in enumerate(intervals):
        if median_int <= 0:
            continue
        if d < 0.5 * median_int:
            amp = float(sig[major_peaks[i]])
            if median_amp > 0 and amp < 0.7 * median_amp:
                at_events.append(int(major_peaks[i]))
    return sorted(list(set(at_events)))

def detect_trigger_delay(signal, major_peaks, fs, delay_threshold_sec=0.15):
    """Detecta trigger delay: tiempo excesivo entre onset y pico."""
    td_events = []
    if len(major_peaks) < 1:
        return td_events
    for p in major_peaks:
        onset = detect_onset(signal, p, fs, back_search_sec=0.6)
        onset_to_peak = float((int(p) - int(onset)) / fs)
        if onset_to_peak > delay_threshold_sec:
            td_events.append({"peak": int(p), "onset": int(onset), "delay_s": float(onset_to_peak)})
    return td_events

def detect_cycling_issues(signal, major_peaks, fs):
    """Detecta ciclado prematuro/tardío comparando Ti."""
    issues = []
    if len(major_peaks) < 2:
        return issues
    cycles = np.diff(np.array(major_peaks)) / float(fs)
    median_cycle = float(np.median(cycles)) if cycles.size>0 else 0.0
    for i in range(len(major_peaks)-1):
        p = int(major_peaks[i])
        onset = detect_onset(signal, p, fs)
        offset = detect_offset(signal, p, fs)
        Ti = (offset - onset) / float(fs) if offset > onset else 0.0
        Ti_ratio = Ti / median_cycle if median_cycle > 0 else 0.0
        if Ti_ratio < 0.6:
            issues.append({"type":"Prematuro","peak":p,"Ti_s":Ti,"Ti_ratio":Ti_ratio})
        elif Ti_ratio > 1.4:
            issues.append({"type":"Tardio","peak":p,"Ti_s":Ti,"Ti_ratio":Ti_ratio})
    return issues

# =========================
# UI / INTERFAZ PRINCIPAL
# =========================

def sidebar_help_markdown():
    md = """
**Ayuda rápida — Parámetros del panel izquierdo**

- **SG Window (odd):** Tamaño de ventana para Savitzky-Golay. Debe ser impar. Ventana mayor = más suavizado (pierde detalle).
- **SG Polyorder:** Orden del polinomio para SG. Ordenes bajos (2-3) son generalmente seguros.
- **Sensibilidad de Trigger (Prominence):** Qué tan 'importante' debe ser un pico para contarse como disparo.
- **Sensibilidad IE (IE prominence scale):** Escala relativa (respecto a la amplitud de picos) para detectar micro-picos en exhalación (Esfuerzos Inefectivos).
"""
    return md

def main():
    # aplicar CSS
    st.markdown(local_css(), unsafe_allow_html=True)

    # Encabezado
    st.title("🫁 Ventilator Lab - Asistente Avanzado")
    st.markdown("""
    **Herramienta de análisis avanzado de asincronías paciente-ventilador.**
    Sube una foto de la pantalla del ventilador para detectar: *Auto-trigger, Esfuerzos Inefectivos y Retrasos de disparo*.
    """)
    st.markdown("---")

    # Sidebar tunables (todos en st.sidebar para consistencia)
    st.sidebar.header("⚙️ Configuración Clínica")
    st.sidebar.markdown("**Desarrollado por:**")
    st.sidebar.markdown("[@JmNunezSilveira](https://instagram.com/JmNunezSilveira)")
    st.sidebar.markdown("---")

    fs = int(st.sidebar.number_input("Frecuencia muestreo estimada (Hz)", min_value=10, value=50, step=1))
    smooth_win = int(st.sidebar.slider("Suavizado (SG Window, impar)", 3, 51, 11, step=2))
    peak_prom = float(st.sidebar.slider("Sensibilidad de Trigger (Prominence)", 0.01, 1.0, 0.2))
    modo = st.sidebar.radio("Tipo de curva detectada", ["Presión (Paw)", "Flujo (Flow)"])

    with st.sidebar.expander("Opciones Avanzadas"):
        smooth_poly = int(st.sidebar.slider("Grado Polinomio (SG)", 1, 5, 3))
        ie_prom_scale = float(st.sidebar.slider("Sensibilidad IE (rel. a pico)", 0.01, 0.2, 0.06))
        st.markdown(sidebar_help_markdown())

    # INPUT PRINCIPAL: Cámara con fallback a uploader
    st.subheader("1. Digitalización de Curva")
    col_input1, col_input2 = st.columns([1, 2])
    with col_input1:
        st.info("Apunta la cámara a la curva y captura (o sube una imagen).")
        img_file_cam = st.camera_input("📸 Capturar Pantalla")
        img_file_up = st.file_uploader("o sube imagen", type=["png", "jpg", "jpeg"])
        # prefer camera if available
        img_file = img_file_cam if img_file_cam is not None else img_file_up

    if img_file is None:
        st.warning("👈 Esperando captura o subida de imagen para iniciar análisis...")
        st.stop()

    # --- PROCESADO IMAGEN ---
    try:
        bytes_data = img_file.getvalue()
        img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            st.error("No se pudo decodificar la imagen. Intente subir otra.")
            st.stop()
    except Exception as e:
        st.error(f"Error leyendo la imagen: {e}")
        st.stop()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # extraer señal
    with st.spinner("Procesando imagen y extrayendo señal..."):
        raw_sig = extract_signal_from_image(gray, col_start_ratio=0.05, col_end_ratio=0.95, invert=True)
        smooth_sig = robust_smooth(raw_sig, window=smooth_win, poly=smooth_poly)
        if smooth_sig.size == 0:
            st.error("Señal extraída vacía. Verifique la calidad de la foto.")
            st.stop()

        # normalizar 0-1 (safe)
        denom = (np.max(smooth_sig) - np.min(smooth_sig))
        if denom <= 1e-9:
            norm_sig = np.zeros_like(smooth_sig)
        else:
            norm_sig = (smooth_sig - np.min(smooth_sig)) / denom

        # detectar picos principales
        prominence_val = max(1e-6, peak_prom * (np.max(norm_sig) if norm_sig.size>0 else 1.0))
        min_dist = max(1, int(0.25 * fs))
        peaks, props = find_peaks(norm_sig, prominence=prominence_val, distance=min_dist)

        # Correr detectores (usando ie_prom_scale del sidebar)
        ie_events = detect_ineffective_efforts(norm_sig, peaks.tolist(), fs, ie_prom_scale=ie_prom_scale)
        at_events = detect_auto_trigger(norm_sig, peaks.tolist(), fs)
        td_events = detect_trigger_delay(norm_sig, peaks.tolist(), fs, delay_threshold_sec=0.15)
        cycling_issues = detect_cycling_issues(norm_sig, peaks.tolist(), fs)

    # --- RESULTADOS ---
    st.subheader("2. Resultados del Análisis")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Ciclos Totales", int(peaks.size))
    col2.metric("Esfuerzos Inefectivos", int(len(ie_events)), delta_color="inverse")
    col3.metric("Auto-trigger", int(len(at_events)), delta_color="inverse")
    col4.metric("Errores de Ciclado", int(len(cycling_issues)), delta_color="inverse")

    # Plot with annotations
    st.markdown("##### Visualización de Eventos")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(norm_sig, color='#00FFFF' if "Flujo" in modo else '#FFFF00', lw=2)
    ax.set_facecolor('black')
    fig.patch.set_facecolor('#F4F6F9')
    ax.axis('off')

    # main peaks
    if peaks.size > 0:
        ax.scatter(peaks, norm_sig[peaks], c='white', s=30, zorder=5, alpha=0.8, label='Trigger')

    # Annotations
    if len(ie_events) > 0:
        ie_idx = np.array(ie_events, dtype=int)
        ie_idx = ie_idx[(ie_idx >= 0) & (ie_idx < len(norm_sig))]
        if ie_idx.size > 0:
            ax.scatter(ie_idx, norm_sig[ie_idx], marker='x', color='orange', s=100, label='Esf. Inefectivo', linewidths=3)

    if len(at_events) > 0:
        at_idx = np.array(at_events, dtype=int)
        at_idx = at_idx[(at_idx >= 0) & (at_idx < len(norm_sig))]
        if at_idx.size > 0:
            ax.scatter(at_idx, norm_sig[at_idx], marker='D', color='magenta', s=80, label='Auto-trigger')

    if len(td_events) > 0:
        for ev in td_events:
            p = int(ev["peak"])
            onset = int(ev["onset"])
            if 0 <= onset < len(norm_sig) and 0 <= p < len(norm_sig):
                ax.plot([onset, p], [norm_sig[onset], norm_sig[p]], color='red', lw=2, linestyle='--')
                ax.text(p, min(1.0, norm_sig[p] + 0.05), f"Retraso {ev['delay_s']:.2f}s", color='red', fontsize=9,
                        bbox=dict(facecolor='black', alpha=0.6))

    for ci in cycling_issues:
        p = int(ci["peak"])
        if 0 <= p < len(norm_sig):
            tag = 'PREMATURO' if ci["type"] == 'Prematuro' else 'TARDIO'
            color = 'orange' if tag == 'PREMATURO' else 'purple'
            ax.text(p, max(0.0, norm_sig[p] - 0.15), tag, color=color, fontsize=8, fontweight='bold',
                    bbox=dict(facecolor='black', alpha=0.6))

    # legend: show only if there are labels
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(loc='upper right', facecolor='#111111', framealpha=0.8)

    st.pyplot(fig)

    # Educational guidance
    st.divider()
    st.subheader("🎓 Interpretación Clínica")
    col_interp1, col_interp2 = st.columns([2, 1])

    with col_interp1:
        if len(td_events) > 0:
            st.error(f"⚠️ **Trigger Delay Detectado:** En {len(td_events)} ciclos el paciente inicia el esfuerzo mucho antes que el ventilador.\n\n*Sugerencia:* Revisa la sensibilidad (Trigger) o busca PEEP intrínseca.")
        elif len(at_events) > 0:
            st.warning("⚠️ **Posible Auto-trigger:** Se detectan ciclos frecuentes de baja amplitud - revisar fugas o oversensitivity del trigger.")
        elif len(ie_events) > 0:
            st.warning("⚠️ **Esfuerzos Inefectivos:** El paciente intenta respirar durante la espiración pero no dispara el ventilador.\n\n*Sugerencia:* Evaluar atrapamiento aéreo o soporte ventilatorio.")
        elif len(cycling_issues) > 0:
            st.info("ℹ️ **Ajuste de Ciclado:** Se detectan discrepancias entre Ti neural y Ti mecánico en algunos ciclos.")
        else:
            st.success("✅ **Sincronía Aceptable:** No se detectaron asincronías heurísticas significativas en este segmento.")

    with col_interp2:
        st.markdown("##### Exportar Datos")
        if st.button("💾 Descargar Informe (.npz)"):
            ts = int(time.time())
            fname = f"vent_lab_{ts}.npz"
            buffer = io.BytesIO()
            np.savez(buffer,
                     raw=raw_sig,
                     smooth=smooth_sig,
                     norm=norm_sig,
                     peaks=peaks,
                     ie_events=np.array(ie_events),
                     at_events=np.array(at_events),
                     td_events=np.array([(e['peak'], e['onset'], e['delay_s']) for e in td_events], dtype=object),
                     cycling=cycling_issues,
                     meta=json.dumps({
                         "fs": fs,
                         "smooth_win": smooth_win,
                         "smooth_poly": smooth_poly,
                         "peak_prom": peak_prom,
                         "ie_prom_scale": ie_prom_scale,
                         "mode": modo,
                         "timestamp": ts
                     })
                     )
            buffer.seek(0)
            st.download_button("Descargar .npz", data=buffer, file_name=fname, mime="application/octet-stream")

    st.markdown("---")
    st.caption("Ventilator Lab Edu v1.0 | Uso educativo exclusivamente. Autor: @JmNunezSilveira")

if __name__ == "__main__":
    main()
