# app.py
import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
import matplotlib
matplotlib.use('Agg')

# --- Configuración de la App ---
st.set_page_config(
    page_title="Ventilator Lab - Asistente Avanzado",
    page_icon="🫁",
    layout="centered"
)

# =========================
# UTIL / DIGITALIZACIÓN
# =========================

def extract_signal_from_image(gray, col_start_ratio=0.1, col_end_ratio=0.9, invert=True):
    """
    Extrae una señal 1D desde una imagen en escala de grises.
    Mejora sobre argmax: utiliza centroid (centroide de intensidad) por columna,
    que funciona mejor con trazos gruesos/ruidosos.
    """
    h, w = gray.shape
    start_col = int(w * col_start_ratio)
    end_col = int(w * col_end_ratio)
    cols = range(start_col, end_col)

    signal = []
    for col in cols:
        col_data = gray[:, col].astype(float)
        # si invert==True asumimos curva brillante sobre fondo oscuro: más brillante = señal
        if invert:
            # invertimos para que "señal" sea mayor en la onda (centroid rises)
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

def robust_smooth(signal, window=11, poly=3):
    """Savitzky-Golay with safe window sizing, plus light gaussian if desired."""
    n = len(signal)
    if n < 5:
        return signal.copy()
    win = int(window)
    if win >= n:
        win = n - 1 if (n - 1) % 2 == 1 else n - 2
    win = max(3, win if win % 2 == 1 else win - 1)
    try:
        sg = savgol_filter(signal, window_length=win, polyorder=min(poly, win-1))
    except Exception:
        sg = signal.copy()
    # small gaussian blur (convolve) for extra smoothing
    kern = np.exp(-0.5 * (np.linspace(-1, 1, 5) ** 2) / (0.2 ** 2))
    kern = kern / np.sum(kern)
    sg2 = np.convolve(sg, kern, mode='same')
    return sg2

# =========================
# ONSET / OFFSET DETECTION
# =========================

def detect_onset(signal, peak_idx, fs, back_search_sec=0.5, slope_thresh_rel=0.2):
    """
    Heurística para detectar inicio de inspiración (onset) antes de un pico:
    Busca el punto donde la derivada supera un umbral relativo.
    """
    n = len(signal)
    back_samples = int(back_search_sec * fs)
    start = max(0, peak_idx - back_samples)
    seg = signal[start:peak_idx+1]
    if seg.size < 3:
        return start
    deriv = np.gradient(seg)
    # threshold relative to max slope in seg
    max_slope = np.max(np.abs(deriv)) + 1e-9
    thr = slope_thresh_rel * max_slope
    # find last index where deriv < thr before the rising begins
    for i in range(len(deriv)-1, -1, -1):
        if deriv[i] <= thr:
            # return global index of onset (next sample)
            onset = start + min(i+1, len(seg)-1)
            return onset
    return start

def detect_offset(signal, peak_idx, fs, forward_search_sec=1.0, slope_thresh_rel=0.05):
    """
    Heurística para detectar fin de inspiración (offset) despues del pico:
    Busca cuando la derivada cruza cero y la señal desciende.
    """
    n = len(signal)
    fwd_samples = int(forward_search_sec * fs)
    end = min(n-1, peak_idx + fwd_samples)
    seg = signal[peak_idx:end+1]
    if seg.size < 3:
        return end
    deriv = np.gradient(seg)
    for i in range(1, len(deriv)):
        # crossing to negative derivative indicating descent/espiración
        if deriv[i] < 0:
            return peak_idx + i
    return end

# =========================
# EVENT DETECTION (B, C, D)
# =========================

def detect_ineffective_efforts(signal, major_peaks, fs):
    """
    Detecta esfuerzos inefectivos: micro picos en la fase espiratoria entre dos picos principales.
    Heurística:
      - Buscar micro-peaks en la porción exhalatoria (entre peaks)
      - Requerir prominencia relativa pequeña (ej. 5-10% de altura_media)
      - Marcar si micro_peak no es seguido por un pico principal en ventana corta
    """
    ie_events = []
    sig = np.asarray(signal, dtype=float)
    if len(major_peaks) < 2 or sig.size == 0:
        return ie_events

    peak_amps = sig[np.array(major_peaks)]
    median_amp = np.median(peak_amps) if peak_amps.size>0 else 1.0
    prominence = max(0.02, 0.06 * median_amp)  # relativo

    for i in range(len(major_peaks)-1):
        s = int(major_peaks[i])
        e = int(major_peaks[i+1])
        if e - s < 5:
            continue
        # define exhalation zone as 30%-90% of interval
        s_zone = s + int((e-s) * 0.3)
        e_zone = s + int((e-s) * 0.9)
        if e_zone <= s_zone:
            continue
        segment = sig[s_zone:e_zone]
        micro_peaks, _ = find_peaks(segment, prominence=prominence, distance=int(0.05*fs))
        for mp in micro_peaks:
            global_idx = s_zone + int(mp)
            # ensure this micro-peak is not close to a main peak
            if not any(abs(global_idx - p) < int(0.2*fs) for p in major_peaks):
                ie_events.append(int(global_idx))
    return sorted(list(set(ie_events)))

def detect_auto_trigger(signal, major_peaks, fs):
    """
    Detecta auto-triggering: series de picos con intervalos muy cortos y amplitudes pequeñas.
    Heurística:
      - intervalos medianos mucho menores a mediana de intervalos (ej < 50%)
      - amplitudes medianas menores a un % de mediana amplitude
    """
    at_events = []
    sig = np.asarray(signal, dtype=float)
    if len(major_peaks) < 4:
        return at_events
    intervals = np.diff(np.array(major_peaks))
    median_int = np.median(intervals)
    for i, d in enumerate(intervals):
        if d < 0.5 * median_int:
            # amplitude check: if peaks nearby are small relative to median amplitude
            amp = sig[major_peaks[i]]
            median_amp = np.median(sig[np.array(major_peaks)])
            if amp < 0.7 * median_amp:
                at_events.append(int(major_peaks[i]))
    return sorted(list(set(at_events)))

def detect_trigger_delay(signal, major_peaks, fs, delay_threshold_sec=0.15):
    """
    Detecta trigger delay: tiempo entre onset (inicio neural) y inicio de aumento significativo del flujo/press.
    Heurística: onset detection (detect_onset) -> compute onset_to_peak time; if > threshold -> delay
    """
    td_events = []
    if len(major_peaks) < 1:
        return td_events
    for p in major_peaks:
        onset = detect_onset(signal, p, fs, back_search_sec=0.6)
        onset_to_peak = (p - onset) / fs
        if onset_to_peak > delay_threshold_sec:
            td_events.append({"peak": int(p), "onset": int(onset), "delay_s": float(onset_to_peak)})
    return td_events

def detect_cycling_issues(signal, major_peaks, fs):
    """
    Detecta ciclado prematuro/tardío comparando Ti (inspiratory time) vs ciclo medio.
    Heurística:
      - Ti = onset->offset per breath
      - Ti_ratio = Ti / (cycle_time)
      - prematuro si Ti_ratio < 0.6 ; tardio si Ti_ratio > 1.4 (umbrales ajustables)
    """
    issues = []
    if len(major_peaks) < 2:
        return issues
    cycles = np.diff(np.array(major_peaks)) / fs
    median_cycle = np.median(cycles)
    for i in range(len(major_peaks)-1):
        p = int(major_peaks[i])
        onset = detect_onset(signal, p, fs)
        offset = detect_offset(signal, p, fs)
        Ti = (offset - onset) / fs if offset > onset else 0.0
        Ti_ratio = Ti / median_cycle if median_cycle > 0 else 0.0
        if Ti_ratio < 0.6:
            issues.append({"type":"Prematuro","peak":p,"Ti_s":Ti,"Ti_ratio":Ti_ratio})
        elif Ti_ratio > 1.4:
            issues.append({"type":"Tardio","peak":p,"Ti_s":Ti,"Ti_ratio":Ti_ratio})
    return issues

# =========================
# UI / INTERFAZ
# =========================

def main():
    st.title("🫁 Ventilator Lab - Avanzado (A-D)")
    st.markdown("Digitalización + IE + Auto-trigger + Trigger Delay + Cycling Issues")

    # Sidebar tunables
    st.sidebar.header("Parámetros")
    fs = int(st.sidebar.number_input("Frecuencia muestreo estimada (Hz)", min_value=10, value=50, step=1))
    smooth_win = st.sidebar.slider("SG window (odd)", 3, 51, 11, step=2)
    smooth_poly = st.sidebar.slider("SG polyorder", 1, 5, 3)
    peak_prom = st.sidebar.slider("Prominence (picos principales)", 0.01, 1.0, 0.2)
    ie_prom_scale = st.sidebar.slider("IE prominence scale (rel to peak amp)", 0.01, 0.2, 0.06)

    modo = st.radio("Tipo de curva", ["Presión (Paw)", "Flujo (Flow)"], horizontal=True)

    img_file = st.camera_input("📸 Capturar Pantalla")

    if img_file is None:
        st.info("Capture o suba una imagen de la pantalla del ventilador.")
        return

    # --- procesado imagen ---
    bytes_data = img_file.getvalue()
    img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # extraer señal (mejor centroid method)
    raw_sig = extract_signal_from_image(gray, col_start_ratio=0.05, col_end_ratio=0.95, invert=True)
    # suavizado robusto
    smooth_sig = robust_smooth(raw_sig, window=smooth_win, poly=smooth_poly)

    # normalizar 0-1 for detection
    norm_sig = (smooth_sig - np.min(smooth_sig)) / (np.max(smooth_sig) - np.min(smooth_sig) + 1e-9)

    # detect main peaks
    prominence_val = max(1e-3, peak_prom * np.max(norm_sig))
    min_dist = int(0.25 * fs)  # 250ms default refractor
    peaks, props = find_peaks(norm_sig, prominence=prominence_val, distance=min_dist)

    # Run detectors
    ie_events = detect_ineffective_efforts(norm_sig, peaks.tolist(), fs)
    at_events = detect_auto_trigger(norm_sig, peaks.tolist(), fs)
    td_events = detect_trigger_delay(norm_sig, peaks.tolist(), fs, delay_threshold_sec=0.15)
    cycling_issues = detect_cycling_issues(norm_sig, peaks.tolist(), fs)

    # Summary KPIs
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Ciclos detectados", len(peaks))
    col2.metric("IE detectados", len(ie_events))
    col3.metric("Auto-trigger", len(at_events))
    col4.metric("Cycling issues", len(cycling_issues))

    # Detailed panels
    with st.expander("Detalles de Eventos (tabla)"):
        st.write("Peaks indices:", peaks.tolist())
        if len(ie_events)>0: st.write("IE indices:", ie_events)
        if len(at_events)>0: st.write("AT indices:", at_events)
        if len(td_events)>0: st.write("Trigger Delay events:", td_events)
        if len(cycling_issues)>0: st.write("Cycling issues:", cycling_issues)

    # Plot with annotations
    fig, ax = plt.subplots(figsize=(10,3))
    ax.plot(norm_sig, color='cyan' if "Flujo" in modo else 'yellow', lw=1.8)
    ax.set_facecolor('black')
    ax.axis('off')

    # main peaks
    if peaks.size>0:
        ax.scatter(peaks, norm_sig[peaks], c='white', s=30, zorder=5, label='Trigger')

    # IE
    if len(ie_events)>0:
        ie_idx = np.array(ie_events, dtype=int)
        ie_idx = ie_idx[(ie_idx>=0)&(ie_idx<len(norm_sig))]
        if ie_idx.size>0:
            ax.scatter(ie_idx, norm_sig[ie_idx], marker='x', color='orange', s=80, label='IE')

    # Auto-trigger
    if len(at_events)>0:
        at_idx = np.array(at_events, dtype=int)
        at_idx = at_idx[(at_idx>=0)&(at_idx<len(norm_sig))]
        if at_idx.size>0:
            ax.scatter(at_idx, norm_sig[at_idx], marker='D', color='magenta', s=70, label='Auto-trigger')

    # Trigger delay: annotate onset -> peak
    if len(td_events)>0:
        for ev in td_events:
            p = int(ev["peak"])
            onset = int(ev["onset"])
            ax.plot([onset,p],[norm_sig[onset],norm_sig[p]], color='red', lw=2, linestyle='--')
            ax.text(p, norm_sig[p]+0.05, f"TD {ev['delay_s']:.2f}s", color='red', fontsize=8)

    # Cycling issues
    for ci in cycling_issues:
        p = int(ci["peak"])
        tag = 'E' if ci["type"]=='Prematuro' else 'L'
        color = 'orange' if tag=='E' else 'purple'
        ax.text(p, norm_sig[p]-0.08, tag, color=color, fontsize=10, fontweight='bold')

    ax.legend(loc='upper right', facecolor='#111111', framealpha=0.6)
    st.pyplot(fig)

    # Educational guidance
    st.divider()
    st.subheader("Interpretación educativa (heurística)")
    if len(td_events)>0:
        st.error(f"Trigger Delay detectado en {len(td_events)} ciclos. Revisar sensibilidad de trigger y condiciones de paciente.")
    elif len(at_events)>0:
        st.warning("Auto-trigger posible: picos frecuentes de baja amplitud — revisar sensibilidad del trigger o fugas.")
    elif len(ie_events)>0:
        st.warning("Esfuerzos inefectivos detectados — paciente hace esfuerzos sin generar ciclo. Revisar soporte y sedación.")
    elif len(cycling_issues)>0:
        st.warning("Ciclado prematuro/tardío detectado en algunos ciclos — ajustar Ti o ciclado.")
    else:
        st.success("No se detectaron asincronías heurísticas significativas en este segmento.")

    # Offer to save results (csv)
    if st.button("Guardar señal + eventos (.npz)"):
        import io, time
        ts = int(time.time())
        fname = f"vent_signal_{ts}.npz"
        np.savez(fname,
                 raw=raw_sig,
                 smooth=smooth_sig,
                 norm=norm_sig,
                 peaks=peaks,
                 ie_events=np.array(ie_events),
                 at_events=np.array(at_events),
                 td_events=np.array([ (e['peak'], e['onset'], e['delay_s']) for e in td_events ], dtype=object),
                 cycling=cycling_issues)
        st.success(f"Guardado {fname} en el directorio local (descargable si corres localmente).")

if __name__ == "__main__":
    main()
