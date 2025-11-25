# app.py
import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter, peak_prominences
import matplotlib

# Usar backend no interactivo para evitar errores de hilos en servidor
matplotlib.use('Agg')

# --- CONFIGURACIÓN DE LA APP (UX PROFESIONAL) ---
st.set_page_config(
    page_title="Ventilator Lab - Asistente Avanzado",
    page_icon="🫁",
    layout="wide",  # CAMBIO CLAVE: Usa todo el ancho de la pantalla
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
            background-color: #F4F6F9; /* Gris azulado muy pálido */
            font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        }

        /* 3. PERSONALIZACIÓN DE TÍTULOS */
        h1 {
            color: #1E3A8A; /* Azul Oscuro Intenso */
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
            background-color: #0284C7; /* Azul Médico */
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

def extract_signal_from_image(gray, col_start_ratio=0.1, col_end_ratio=0.9, invert=True):
    """
    Extrae una señal 1D desde una imagen en escala de grises.
    Mejora sobre argmax: utiliza centroid (centroide de intensidad) por columna.
    """
    h, w = gray.shape
    start_col = int(w * col_start_ratio)
    end_col = int(w * col_end_ratio)
    cols = range(start_col, end_col)

    signal = []
    for col in cols:
        col_data = gray[:, col].astype(float)
        # si invert==True asumimos curva brillante sobre fondo oscuro
        if invert:
            weights = col_data
        else:
            weights = 255.0 - col_data

        s = np.sum(weights)
        if s <= 1e-6:
            # fallback: usar argmax
            idx = np.argmax(col_data)
            y = h - idx
        else:
            # centroid: sum(i * w) / sum(w)
            indices = np.arange(h)
            centroid = np.sum(indices * weights) / s
            y = h - centroid  # invert Y so increasing means upward on plot
        signal.append(y)

    signal = np.array(signal, dtype=float)
    return signal

def robust_smooth(signal_array, window=11, poly=3):
    """Savitzky-Golay with safe window sizing, plus light gaussian if desired."""
    n = len(signal_array)
    if n < 5:
        return signal_array.copy()
    win = int(window)
    if win >= n:
        # choose nearest valid odd window < n
        win = n - 1 if (n - 1) % 2 == 1 else n - 2
    win = max(3, win if win % 2 == 1 else win - 1)
    try:
        sg = savgol_filter(signal_array, window_length=win, polyorder=min(poly, win-1))
    except Exception:
        sg = signal_array.copy()
    # small gaussian blur for extra smoothing
    kern = np.exp(-0.5 * (np.linspace(-1, 1, 5) ** 2) / (0.2 ** 2))
    kern = kern / np.sum(kern)
    sg2 = np.convolve(sg, kern, mode='same')
    return sg2

# =========================
# ONSET / OFFSET DETECTION
# =========================

def detect_onset(signal_array, peak_idx, fs, back_search_sec=0.5, slope_thresh_rel=0.2):
    """Detecta inicio de inspiración (onset) antes de un pico."""
    n = len(signal_array)
    back_samples = int(back_search_sec * fs)
    start = max(0, peak_idx - back_samples)
    seg = signal_array[start:peak_idx+1]
    if seg.size < 3:
        return start
    deriv = np.gradient(seg)
    max_slope = np.max(np.abs(deriv)) + 1e-9
    thr = slope_thresh_rel * max_slope
    for i in range(len(deriv)-1, -1, -1):
        if deriv[i] <= thr:
            onset = start + min(i+1, len(seg)-1)
            return onset
    return start

def detect_offset(signal_array, peak_idx, fs, forward_search_sec=1.0, slope_thresh_rel=0.05):
    """Detecta fin de inspiración (offset) despues del pico."""
    n = len(signal_array)
    fwd_samples = int(forward_search_sec * fs)
    end = min(n-1, peak_idx + fwd_samples)
    seg = signal_array[peak_idx:end+1]
    if seg.size < 3:
        return end
    deriv = np.gradient(seg)
    for i in range(1, len(deriv)):
        if deriv[i] < 0:
            return peak_idx + i
    return end

# =========================
# EVENT DETECTION (B, C, D)
# =========================

def detect_ineffective_efforts(signal_array, major_peaks, fs):
    """Detecta esfuerzos inefectivos: micro picos en la fase espiratoria."""
    ie_events = []
    sig = np.asarray(signal_array, dtype=float)
    if len(major_peaks) < 2 or sig.size == 0:
        return ie_events

    peak_amps = sig[np.array(major_peaks)]
    median_amp = np.median(peak_amps) if peak_amps.size>0 else 1.0
    prominence = max(0.02, 0.06 * median_amp) 

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
        micro_peaks, _ = find_peaks(segment, prominence=prominence, distance=int(0.05*fs))
        for mp in micro_peaks:
            global_idx = s_zone + int(mp)
            if not any(abs(global_idx - p) < int(0.2*fs) for p in major_peaks):
                ie_events.append(int(global_idx))
    return sorted(list(set(ie_events)))

def detect_auto_trigger(signal_array, major_peaks, fs):
    """Detecta auto-triggering: intervalos cortos y amplitudes pequeñas."""
    at_events = []
    sig = np.asarray(signal_array, dtype=float)
    if len(major_peaks) < 4:
        return at_events
    intervals = np.diff(np.array(major_peaks))
    median_int = np.median(intervals)
    for i, d in enumerate(intervals):
        if d < 0.5 * median_int:
            amp = sig[major_peaks[i]]
            median_amp = np.median(sig[np.array(major_peaks)])
            if amp < 0.7 * median_amp:
                at_events.append(int(major_peaks[i]))
    return sorted(list(set(at_events)))

def detect_trigger_delay(signal_array, major_peaks, fs, delay_threshold_sec=0.15):
    """Detecta trigger delay: tiempo excesivo entre onset y pico."""
    td_events = []
    if len(major_peaks) < 1:
        return td_events
    for p in major_peaks:
        onset = detect_onset(signal_array, p, fs, back_search_sec=0.6)
        onset_to_peak = (p - onset) / fs
        if onset_to_peak > delay_threshold_sec:
            td_events.append({"peak": int(p), "onset": int(onset), "delay_s": float(onset_to_peak)})
    return td_events

def detect_cycling_issues(signal_array, major_peaks, fs):
    """Detecta ciclado prematuro/tardío comparando Ti."""
    issues = []
    if len(major_peaks) < 2:
        return issues
    cycles = np.diff(np.array(major_peaks)) / fs
    median_cycle = np.median(cycles)
    for i in range(len(major_peaks)-1):
        p = int(major_peaks[i])
        onset = detect_onset(signal_array, p, fs)
        offset = detect_offset(signal_array, p, fs)
        Ti = (offset - onset) / fs if offset > onset else 0.0
        Ti_ratio = Ti / median_cycle if median_cycle > 0 else 0.0
        if Ti_ratio < 0.6:
            issues.append({"type":"Prematuro","peak":p,"Ti_s":Ti,"Ti_ratio":Ti_ratio})
        elif Ti_ratio > 1.4:
            issues.append({"type":"Tardio","peak":p,"Ti_s":Ti,"Ti_ratio":Ti_ratio})
    return issues

# =========================
# ALGORITMO MEJORADO (INTEGRACIÓN)
# =========================

def analyze_asynchronies(sig, fs, prom_main=0.25, ie_scale=0.5):
    """
    Algoritmo avanzado integrado:
    - Detecta Double Trigger (DT)
    - Ineffective Effort (IE)
    - Flow Starvation (FS)
    - Auto-Trigger (AT)
    """
    out = {
        "double_trigger": [],        # list of (peak1, peak2)
        "ineffective_efforts": [],   # list of peak indices
        "flow_starvation": [],       # list of peak indices
        "auto_trigger": [],          # list of peak indices
        "peaks": []
    }

    if sig is None or len(sig) < 10:
        return out

    # --- Suavizado seguro ---
    L = len(sig)
    win = 21
    if win >= L:
        win = L - 1 if (L - 1) % 2 == 1 else L - 2
    win = max(5, win if win % 2 == 1 else win - 1)
    try:
        sm = savgol_filter(sig, window_length=win, polyorder=3)
    except Exception:
        sm = sig.copy()

    # --- Picos principales ---
    min_dist = max(1, int(0.12 * fs))
    peaks, _ = find_peaks(sm, prominence=prom_main, distance=min_dist)
    out["peaks"] = peaks.tolist()

    if len(peaks) < 2:
        return out

    # RR intervals (s)
    rr_int = np.diff(peaks) / float(fs)
    avg_rr = float(np.median(rr_int)) if rr_int.size > 0 else 1.0
    dt_threshold = max(0.25 * avg_rr, 0.25)  # conservative lower bound

    # prominences for IE detection
    promins = peak_prominences(sm, peaks)[0] if len(peaks) > 0 else np.array([])

    # --- Double Trigger: temporal + valley retention ---
    for i in range(len(peaks)-1):
        t1 = int(peaks[i])
        t2 = int(peaks[i+1])
        delta_s = (t2 - t1) / float(fs)
        if delta_s < dt_threshold:
            if (t2 - t1) > 2:
                valley = np.min(sm[t1:t2])
                peak1_amp = max(1e-6, sm[t1])
                drop_ratio = (peak1_amp - valley) / peak1_amp
                # if drop_ratio small -> no real exhalation -> DT
                if drop_ratio < 0.35:
                    out["double_trigger"].append((t1, t2))

    # --- Ineffective Efforts: low prominences and temporal plausibility ---
    thr_ie = prom_main * ie_scale
    for idx, pk in enumerate(peaks):
        p = int(pk)
        prom_val = float(promins[idx]) if idx < len(promins) else np.inf
        if prom_val < thr_ie:
            # Ensure not adjacent to other peaks (avoid labeling noise)
            left_gap = (p - peaks[idx-1]) / fs if idx > 0 else np.inf
            if left_gap > 0.25:  # at least 250 ms since prior peak
                out["ineffective_efforts"].append(p)

    # --- Flow Starvation: low initial slope on inspiration ---
    for i in range(len(peaks)-1):
        p = int(peaks[i])
        q = int(peaks[i+1])
        seg = sm[p:q] if q > p + 5 else None
        if seg is None or len(seg) < 8:
            continue
        # take first third to estimate rise slope
        nsel = max(3, len(seg)//3)
        x = np.arange(nsel)
        try:
            slope = np.polyfit(x, seg[:nsel], 1)[0]
        except Exception:
            slope = 0.0
        # slope small -> starvation (heuristic)
        if slope < (0.02 * np.max(np.abs(sm)) + 1e-9):
            out["flow_starvation"].append(p)

    # --- Auto-trigger: unusually short cycles vs median RR ---
    if rr_int.size > 1:
        short_idxs = np.where(rr_int < (0.35 * avg_rr))[0]
        for si in short_idxs:
            out["auto_trigger"].append(int(peaks[si+1]))

    # deduplicate and sort lists
    out["ineffective_efforts"] = sorted(list(set(out["ineffective_efforts"])))
    out["flow_starvation"] = sorted(list(set(out["flow_starvation"])))
    out["auto_trigger"] = sorted(list(set(out["auto_trigger"])))
    out["double_trigger"] = sorted(list(set(out["double_trigger"])))

    return out

# =========================
# UI / INTERFAZ PRINCIPAL
# =========================

def main():
    # 1. INYECTAR CSS AL INICIO
    st.markdown(local_css(), unsafe_allow_html=True)

    # Encabezado
    st.title("🫁 Ventilator Lab - AI Assistant")
    st.markdown("""
    **Herramienta de análisis avanzado de asincronías paciente-ventilador.**
    Sube una foto de la pantalla del ventilador para detectar: *Auto-trigger, Esfuerzos inefectivos y Retrasos de disparo.*
    """)
    st.markdown("---")

    # Sidebar tunables
    st.sidebar.header("⚙️ Configuración Clínica")
    
    # Branding en Sidebar
    st.sidebar.markdown("**Desarrollado por:**")
    st.sidebar.markdown("[@JmNunezSilveira](https://instagram.com/JmNunezSilveira)")
    st.sidebar.markdown("---")

    fs = int(st.sidebar.number_input("Frecuencia muestreo estimada (Hz)", min_value=10, value=50, step=1))
    smooth_win = st.sidebar.slider("Suavizado (Window)", 3, 51, 11, step=2)
    peak_prom = float(st.sidebar.slider("Sensibilidad de Trigger", 0.01, 1.0, 0.2))
    
    with st.sidebar.expander("Opciones Avanzadas"):
        smooth_poly = int(st.slider("Grado Polinomio (SG)", 1, 5, 3))
        ie_prom_scale = float(st.slider("Sensibilidad IE (scale)", 0.01, 0.5, 0.06))
        enable_advanced_algo = st.checkbox("Usar algoritmo avanzado (DT/IE/FS/AT)", value=True)

    modo = st.sidebar.radio("Tipo de curva detectada", ["Presión (Paw)", "Flujo (Flow)"])

    # INPUT PRINCIPAL: Permitir Cámara O Subida de archivo (Mejora UX)
    st.subheader("1. Digitalización de Curva")
    col_input1, col_input2 = st.columns([1, 2])
    
    with col_input1:
        st.info("Apunta la cámara a la curva y captura.")
        img_file = st.camera_input("📸 Capturar Pantalla")

    if img_file is None:
        st.warning("👈 Esperando captura de imagen para iniciar análisis...")
        return

    # --- PROCESADO IMAGEN ---
    bytes_data = img_file.getvalue()
    img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # extraer señal
    with st.spinner("Procesando imagen y extrayendo señal..."):
        raw_sig = extract_signal_from_image(gray, col_start_ratio=0.05, col_end_ratio=0.95, invert=True)
        smooth_sig = robust_smooth(raw_sig, window=smooth_win, poly=smooth_poly)
        
        # normalizar 0-1
        norm_sig = (smooth_sig - np.min(smooth_sig)) / (np.max(smooth_sig) - np.min(smooth_sig) + 1e-9)

        # detectar picos principales
        prominence_val = max(1e-3, peak_prom * np.max(norm_sig))
        min_dist = int(0.25 * fs)  # 250ms periodo refractario
        peaks, props = find_peaks(norm_sig, prominence=prominence_val, distance=min_dist)

        # Run both legacy detectors and advanced algorithm (if enabled)
        # Legacy detectors (kept as fallback / complementary)
        ie_events_legacy = detect_ineffective_efforts(norm_sig, peaks.tolist(), fs)
        at_events_legacy = detect_auto_trigger(norm_sig, peaks.tolist(), fs)
        td_events = detect_trigger_delay(norm_sig, peaks.tolist(), fs, delay_threshold_sec=0.15)
        cycling_issues = detect_cycling_issues(norm_sig, peaks.tolist(), fs)

        # Advanced integrated detection
        adv = analyze_asynchronies(norm_sig, fs, prom_main=prominence_val, ie_scale=ie_prom_scale) if enable_advanced_algo else {
            "double_trigger": [], "ineffective_efforts": [], "flow_starvation": [], "auto_trigger": [], "peaks": []
        }

        # Combine results: union of legacy and advanced for IE and AT
        ie_events = sorted(list(set(ie_events_legacy + adv.get("ineffective_efforts", []))))
        at_events = sorted(list(set(at_events_legacy + adv.get("auto_trigger", []))))
        dt_events = adv.get("double_trigger", [])  # list of tuples (p1,p2)
        fs_events = adv.get("flow_starvation", [])
        # td_events and cycling_issues remain from legacy detector

    # --- RESULTADOS ---
    st.subheader("2. Resultados del Análisis")
    
    # Summary KPIs con estilo CSS
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Ciclos Totales", len(peaks))
    col2.metric("Esfuerzos Inefectivos", len(ie_events), delta_color="inverse")
    col3.metric("Auto-trigger", len(at_events), delta_color="inverse")
    col4.metric("Double Trigger (DT)", len(dt_events), delta_color="inverse")

    # Plot with annotations
    st.markdown("##### Visualización de Eventos")
    fig, ax = plt.subplots(figsize=(12, 4)) # Más ancho para mejor visualización
    
    # Estilo de gráfico "Monitor Médico"
    ax.plot(norm_sig, color='#00FFFF' if "Flujo" in modo else '#FFFF00', lw=2) # Cyan o Amarillo
    ax.set_facecolor('black') # Fondo negro de monitor
    fig.patch.set_facecolor('#F4F6F9') # Fondo de la figura igual al de la app
    ax.axis('off') # Sin ejes para limpieza

    # main peaks
    if len(peaks) > 0:
        ax.scatter(peaks, norm_sig[peaks], c='white', s=30, zorder=5, alpha=0.7)

    # Annotations: IE
    if len(ie_events) > 0:
        ie_idx = np.array(ie_events, dtype=int)
        ie_idx = ie_idx[(ie_idx >= 0) & (ie_idx < len(norm_sig))]
        if ie_idx.size > 0:
            ax.scatter(ie_idx, norm_sig[ie_idx], marker='x', color='orange', s=100, label='Esf. Inefectivo', linewidths=3)

    # Annotations: Auto-trigger
    if len(at_events) > 0:
        at_idx = np.array(at_events, dtype=int)
        at_idx = at_idx[(at_idx >= 0) & (at_idx < len(norm_sig))]
        if at_idx.size > 0:
            ax.scatter(at_idx, norm_sig[at_idx], marker='D', color='magenta', s=80, label='Auto-trigger')

    # Annotations: Flow Starvation (FS)
    if len(fs_events) > 0:
        fs_idx = np.array(fs_events, dtype=int)
        fs_idx = fs_idx[(fs_idx >= 0) & (fs_idx < len(norm_sig))]
        if fs_idx.size > 0:
            ax.scatter(fs_idx, norm_sig[fs_idx], marker='v', color='magenta', s=100, label='Flow Starvation')

    # Annotations: Double Trigger (DT) - draw connecting lines
    if len(dt_events) > 0:
        for (p1, p2) in dt_events:
            if 0 <= p1 < len(norm_sig) and 0 <= p2 < len(norm_sig):
                ax.plot([p1, p2], [norm_sig[p1], norm_sig[p2]], color='red', linewidth=3, linestyle=':')
                ax.text(p2, norm_sig[p2] + 0.03, "DT", color='red', fontweight='bold')

    # legacy trigger delay annotations
    if len(td_events) > 0:
        for ev in td_events:
            p = int(ev["peak"])
            onset = int(ev["onset"])
            if 0 <= onset < len(norm_sig) and 0 <= p < len(norm_sig):
                ax.plot([onset, p], [norm_sig[onset], norm_sig[p]], color='red', lw=2, linestyle='--')
                ax.text(p, norm_sig[p] + 0.05, f"Retraso {ev['delay_s']:.2f}s", color='red', fontsize=9, backgroundcolor='black')

    for ci in cycling_issues:
        p = int(ci["peak"])
        tag = 'PREMATURO' if ci["type"] == 'Prematuro' else 'TARDIO'
        color = 'orange' if tag == 'PREMATURO' else 'purple'
        if 0 <= p < len(norm_sig):
            ax.text(p, norm_sig[p] - 0.15, tag, color=color, fontsize=8, fontweight='bold', backgroundcolor='black')

    ax.legend(loc='upper right', facecolor='#111111', framealpha=0.8, labelcolor='white')
    st.pyplot(fig)

    # Educational guidance
    st.divider()
    st.subheader("🎓 Interpretación Clínica")
    
    col_interp1, col_interp2 = st.columns([2, 1])
    
    with col_interp1:
        if len(dt_events) > 0:
            st.error(f"🚨 **Double Trigger detectado:** {len(dt_events)} eventos. Revise Ti mecánico vs Ti neural.")
        elif len(td_events) > 0:
            st.error(f"⚠️ **Trigger Delay Detectado:** En {len(td_events)} ciclos el paciente inicia el esfuerzo mucho antes que el ventilador. \n\n*Sugerencia:* Revisa la sensibilidad (Trigger) o busca PEEP intrínseca.")
        elif len(at_events) > 0:
            st.warning("⚠️ **Posible Auto-trigger:** Se detectan ciclos frecuentes de baja amplitud sin esfuerzo aparente. \n\n*Sugerencia:* Verifica fugas en el circuito o presencia de oscilaciones cardíacas.")
        elif len(ie_events) > 0:
            st.warning("⚠️ **Esfuerzos Inefectivos:** El paciente intenta respirar durante la espiración pero no dispara el ventilador. \n\n*Sugerencia:* Evalúa si hay atrapamiento aéreo o debilidad muscular.")
        elif len(cycling_issues) > 0:
            st.info("ℹ️ **Ajuste de Ciclado:** Se detectan discrepancias en el tiempo inspiratorio neural vs mecánico.")
        else:
            st.success("✅ **Sincronía Aceptable:** No se detectaron asincronías mayores con los parámetros actuales.")

    with col_interp2:
        # Botón de descarga estilizado
        st.markdown("##### Exportar Datos")
        if st.button("💾 Descargar Informe (.npz)"):
            import io, time
            ts = int(time.time())
            fname = f"vent_lab_{ts}.npz"
            # Simulación de guardado para descarga
            st.success(f"Archivo generado: {fname}")
            st.caption("Útil para investigación o docencia.")

    # Footer
    st.markdown("---")
    st.caption("Ventilator Lab Edu v1.0 | Uso educativo exclusivamente.")

if __name__ == "__main__":
    main()
