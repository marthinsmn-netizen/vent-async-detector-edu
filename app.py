"""
Phase 5: Precision Algorithms - Ventilator Asynchrony Detection System
Filename: app.py

Descripción General:
    Motor de análisis para la 'Fase 5: Algoritmos de Precisión'.
"""
import logging
import numpy as np
from scipy.signal import find_peaks
from flask import Flask, request, jsonify

# Configuración del sistema de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

class AlgorithmConfig:
    SAMPLING_RATE_HZ = 50
    DEFICIT_AREA_THRESHOLD = 5.0
    DT_TIME_WINDOW_SEC = 1.5
    VALLEY_RETENTION_RATIO = 0.10
    MIN_FLOW_TRIGGER = 2.0

class WaveformProcessor:
    @staticmethod
    def smooth_signal(signal_array, window_size=3):
        window_size = max(1, int(window_size))
        window = np.ones(window_size) / window_size
        return np.convolve(np.asarray(signal_array, dtype=float), window, mode='same')

    @staticmethod
    def calculate_area_of_deficit(time_vector, flow_vector, breath_type='VC'):
        time_vector = np.asarray(time_vector, dtype=float)
        flow_vector = np.asarray(flow_vector, dtype=float)

        if time_vector.size == 0 or flow_vector.size == 0 or time_vector.size != flow_vector.size:
            return 0.0, 0.0

        max_flow = float(np.max(flow_vector))
        max_idx = int(np.argmax(flow_vector))

        reference_curve = np.zeros_like(flow_vector)

        if breath_type == 'VC':
            reference_curve[:] = max_flow
        elif breath_type == 'PC':
            end_flow = float(flow_vector[-1])
            denom = max(1, (len(flow_vector) - 1 - max_idx))
            slope = (end_flow - max_flow) / denom if denom != 0 else 0.0
            reference_curve[:max_idx] = flow_vector[:max_idx]
            for i in range(max_idx, len(flow_vector)):
                reference_curve[i] = max_flow + slope * (i - max_idx)
        else:
            # fallback: treat as VC
            reference_curve[:] = max_flow

        deficit_vector = reference_curve - flow_vector
        deficit_vector[deficit_vector < 0] = 0.0

        # integrate using trapezoidal rule (units consistent with time_vector)
        area_deficit = float(np.trapz(deficit_vector, x=time_vector))
        total_reference_area = float(np.trapz(reference_curve, x=time_vector))

        return area_deficit, total_reference_area

    @staticmethod
    def detect_valley_double_trigger(time_vector, flow_vector):
        time_vector = np.asarray(time_vector, dtype=float)
        flow_vector = np.asarray(flow_vector, dtype=float)

        if flow_vector.size == 0 or time_vector.size == 0 or time_vector.size != flow_vector.size:
            return False, {"reason": "Invalid input arrays"}

        smoothed_flow = WaveformProcessor.smooth_signal(flow_vector, window_size=5)

        # find peaks with modest prominence/distance (tunable)
        peaks, properties = find_peaks(smoothed_flow, prominence=0.1 * np.max(np.abs(smoothed_flow) + 1e-6), distance=3)

        if peaks.size < 2:
            return False, {"reason": "Picos insuficientes (<2)"}

        # take first two significant peaks (could be improved using prominences)
        p1_idx = int(peaks[0])
        p2_idx = int(peaks[1])

        # ensure indices ordered
        if p2_idx <= p1_idx:
            return False, {"reason": "Picos no ordenados"}

        # temporal check
        time_diff = float(time_vector[p2_idx] - time_vector[p1_idx])
        if time_diff > AlgorithmConfig.DT_TIME_WINDOW_SEC:
            return False, {"reason": "Picos demasiado separados", "time_diff": time_diff}

        # segment between peaks
        segment = smoothed_flow[p1_idx:p2_idx+1]  # include p2 point
        if segment.size == 0:
            return False, {"reason": "Segmento entre picos vacío"}

        valley_value = float(np.min(segment))
        p1_amp = float(smoothed_flow[p1_idx])

        ratio = 0.0
        is_double_trigger = False
        if p1_amp > 0:
            ratio = valley_value / p1_amp
            if ratio > AlgorithmConfig.VALLEY_RETENTION_RATIO:
                is_double_trigger = True

        return is_double_trigger, {
            "valley_flow": float(valley_value),
            "peak1_flow": float(p1_amp),
            "valley_retention_ratio": float(ratio),
            "threshold_applied": AlgorithmConfig.VALLEY_RETENTION_RATIO,
            "time_diff": time_diff,
            "diagnosis": "Breath Stacking" if is_double_trigger else "Normal Cycling"
        }

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "Phase 5 Algorithms Operational", "version": "5.0.1"}), 200

@app.route('/analyze/waveform', methods=['POST'])
def analyze_waveform():
    try:
        data = request.get_json(force=True)
        if not data or 'time' not in data or 'flow' not in data:
            return jsonify({"error": "Datos de forma de onda incompletos (requiere 'time', 'flow')"}), 400

        time_vec = np.asarray(data['time'], dtype=float)
        flow_vec = np.asarray(data['flow'], dtype=float)
        breath_mode = str(data.get('mode', 'VC')).upper()

        # calculate area of deficit
        deficit_area, ref_area = WaveformProcessor.calculate_area_of_deficit(time_vec, flow_vec, breath_mode)

        starvation_severity = "None"
        if deficit_area > AlgorithmConfig.DEFICIT_AREA_THRESHOLD:
            starvation_severity = "Severe" if deficit_area > (AlgorithmConfig.DEFICIT_AREA_THRESHOLD * 2) else "Moderate"

        # detect double trigger
        is_dt, dt_metrics = WaveformProcessor.detect_valley_double_trigger(time_vec, flow_vec)

        # sanitize dt_metrics numeric rounding where possible
        dt_metrics_safe = {}
        for k, v in dt_metrics.items():
            try:
                dt_metrics_safe[k] = float(round(v, 4)) if isinstance(v, (int, float, np.floating, np.integer)) else v
            except Exception:
                dt_metrics_safe[k] = v

        response = {
            "analysis_metadata": {
                "phase": "Phase 5: Precision Algorithms",
                "algorithm_version": "5.0.1"
            },
            "metrics": {
                "flow_starvation": {
                    "detected": deficit_area > AlgorithmConfig.DEFICIT_AREA_THRESHOLD,
                    "area_of_deficit_value": float(round(deficit_area, 6)),
                    "reference_area": float(round(ref_area, 6)),
                    "deficit_percentage": float(round((deficit_area / ref_area) * 100, 2)) if ref_area > 0 else 0.0,
                    "severity": starvation_severity,
                    "description": "Integral de la diferencia entre flujo ideal y real (Concavidad)"
                },
                "double_trigger": {
                    "detected": bool(is_dt),
                    "valley_metrics": dt_metrics_safe,
                    "diagnosis": dt_metrics_safe.get("diagnosis", ""),
                    "description": "Detección basada en retención de flujo en el valle inspiratorio"
                }
            }
        }

        return jsonify(response), 200

    except Exception as e:
        logging.exception("Error en Procesamiento de Onda")
        return jsonify({"error": "Error Interno de Procesamiento", "details": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
