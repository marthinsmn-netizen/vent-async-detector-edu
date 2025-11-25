"""
Phase 5: Precision Algorithms - Ventilator Asynchrony Detection System
Filename: app.py

Descripción General:
    Este módulo implementa el motor de análisis para la 'Fase 5: Algoritmos de Precisión'.
    Su objetivo es la detección y cuantificación morfológica de asincronías ventilatorias,
    superando la detección binaria tradicional.
    
    Mejoras Clave Implementadas:
    1. 'Área de Déficit' (Flow Starvation): Calcula la integral de la diferencia entre
       la curva de flujo/presión ideal y la real para cuantificar el "Hambre de Aire".
    2. 'Umbral de Valle' (Double Trigger): Analiza la profundidad del flujo entre dos
       picos consecutivos para diferenciar entre taquipnea y apilamiento de respiraciones (breath stacking).

    Fundamentos Técnicos y Científicos:
    - Integración Numérica: Utiliza la Regla del Trapecio (numpy.trapz) para el cálculo
      de áreas bajo curvas discretas.
    - Procesamiento de Señales: Detección de picos y valles mediante Scipy, aplicando
      lógica de umbrales dinámicos inspirada en sistemas LIDAR y análisis de ondas.
    - Fisiopatología: Definiciones basadas en la cinética de la inanición de flujo [7]
      y el disparo doble.

Autor: Experto en Procesamiento de Señales Biomédicas
Fase: 5 (Algoritmos de Precisión)
"""

import logging
import numpy as np
from scipy.signal import find_peaks
from flask import Flask, request, jsonify

# Configuración del sistema de logging para trazabilidad clínica y depuración
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

class AlgorithmConfig:
    """
    Clase de Configuración para los Algoritmos de la Fase 5.
    
    Estos parámetros calibran la sensibilidad de los algoritmos de precisión.
    En una implementación futura, estos valores podrían ser dinámicos basados en el
    Peso Corporal Ideal (IBW) del paciente o la compliance pulmonar.
    """
    # Frecuencia de muestreo típica de ventiladores modernos (50Hz o 100Hz) 
    SAMPLING_RATE_HZ = 50
    
    # --- Parámetros para Área de Déficit (Flow Starvation) ---
    # Un área de déficit > este valor indica un trabajo respiratorio significativo.
    # Unidad: (L/min * seg) -> Proxi de volumen perdido o esfuerzo no asistido.
    # Se establece un umbral base conservador para evitar falsos positivos por ruido.
    DEFICIT_AREA_THRESHOLD = 5.0 
    
    # --- Parámetros para Umbral de Valle (Double Trigger) ---
    # Ventana de tiempo máxima entre picos para considerar un posible apilamiento.
    #  sugiere que el DT ocurre en un intervalo corto (<50% tiempo insp).
    DT_TIME_WINDOW_SEC = 1.5
    
    # RATIO DE RETENCIÓN DE VALLE (The Valley Threshold)
    # Si el flujo en el valle (entre dos picos) se mantiene por ENCIMA de este porcentaje
    # del primer pico, indica que la exhalación fue incompleta o inexistente.
    # 0.10 significa que el flujo solo bajó al 10% del pico antes de volver a subir.
    VALLEY_RETENTION_RATIO = 0.10  # 10%
    
    # Flujo mínimo absoluto para considerar un evento como inspiración válida (L/min)
    MIN_FLOW_TRIGGER = 2.0

class WaveformProcessor:
    """
    Motor central de procesamiento de señales.
    Encapsula la lógica matemática y física para la Fase 5.
    """

    @staticmethod
    def smooth_signal(signal_array, window_size=3):
        """
        Aplica un suavizado de media móvil para reducir el ruido del sensor.
        Crucial para evitar que fluctuaciones menores sean detectadas como picos o valles falsos.
        """
        window = np.ones(window_size) / window_size
        # 'same' devuelve un array del mismo tamaño que la entrada
        return np.convolve(signal_array, window, mode='same')

    @staticmethod
    def calculate_area_of_deficit(time_vector, flow_vector, breath_type='VC'):
        """
        Calcula el 'Área de Déficit' representando la Inanición de Flujo (Flow Starvation).
        
        Teoría:
            La inanición de flujo se manifiesta como una concavidad ('scooping') en la onda.
            Matemáticamente: Déficit = Integral(Curva_Referencia - Curva_Real).
            
        Método:
            Utiliza integración trapezoidal compuesta (numpy.trapz).
            Esta aproximación es robusta para datos muestreados equidistantemente.
            
        Argumentos:
            time_vector (np.array): Vector de tiempo.
            flow_vector (np.array): Vector de flujo inspiratorio.
            breath_type (str): 'VC' (Volumen Control - Referencia Cuadrada) o 
                               'PC' (Presión Control - Referencia Desacelerante).
                               
        Retorna:
            float: El área de déficit calculada (magnitud del hambre de aire).
            float: El área total de referencia (para cálculo de porcentaje relativo).
        """
        # 1. Identificación de Puntos Críticos
        # Encontramos el flujo máximo para escalar la curva de referencia.
        max_flow = np.max(flow_vector)
        max_idx = np.argmax(flow_vector)
        
        # 2. Generación de la Curva de Referencia (El "Ideal")
        reference_curve = np.zeros_like(flow_vector)
        
        if breath_type == 'VC':
            # Volumen Control: El flujo ideal es una Onda Cuadrada (flujo constante).
            # Referencia = Flujo Máximo mantenido hasta el final de la inspiración.
            reference_curve[:] = max_flow
            
        elif breath_type == 'PC':
            # Presión Control: El flujo ideal es Desacelerante (Exponencial o Lineal).
            # Aproximamos una referencia lineal desde el Pico hasta el Flujo Final.
            # Cualquier caída por debajo de esta línea recta indica una concavidad anómala.
            
            end_flow = flow_vector[-1]
            # Calculamos la pendiente de desaceleración ideal
            slope = (end_flow - max_flow) / (len(flow_vector) - max_idx) if len(flow_vector) > max_idx else 0
            
            # Antes del pico, asumimos que la subida es correcta (sin déficit inicial)
            reference_curve[:max_idx] = flow_vector[:max_idx]
            
            # Después del pico, proyectamos la línea de desaceleración ideal
            for i in range(max_idx, len(flow_vector)):
                reference_curve[i] = max_flow + slope * (i - max_idx)

        # 3. Cálculo del Vector de Déficit
        # Solo nos interesa donde Referencia > Real (Concavidad/Hambre de aire).
        # Si Real > Referencia (Overshoot), el déficit es 0 para ese punto.
        deficit_vector = reference_curve - flow_vector
        deficit_vector[deficit_vector < 0] = 0  # Clamp de valores negativos
        
        # 4. Integración Numérica (Trapezoidal Rule)
        #  y  validan np.trapz para el cálculo de áreas bajo curvas discretas.
        # Pasamos 'x=time_vector' para asegurar que el área tenga unidades físicas (Litros).
        area_deficit = np.trapz(deficit_vector, x=time_vector)
        total_reference_area = np.trapz(reference_curve, x=time_vector)
        
        return area_deficit, total_reference_area

    @staticmethod
    def detect_valley_double_trigger(time_vector, flow_vector):
        """
        Implementa la lógica de 'Umbral de Valle' para detección de Disparo Doble.
        
        Teoría:
            El Disparo Doble consiste en dos esfuerzos apilados sin exhalación completa.
            Característica Clave: El flujo desciende (formando un Valle) pero NO retorna
            a la línea base (o fase espiratoria) antes de subir nuevamente.
            
        Algoritmo:
            1. Encontrar Picos significativos.
            2. Si hay >1 pico cercano, analizar el Valle (mínimo) entre ellos.
            3. Aplicar Umbral de Valle: ¿Es el valle suficientemente profundo?
            
        Retorna:
            bool: True si se detecta Disparo Doble (Breath Stacking).
            dict: Métricas detalladas (profundidad del valle, ratios).
        """
        # Suavizar señal para evitar falsos positivos por ruido
        smoothed_flow = WaveformProcessor.smooth_signal(flow_vector)
        
        # 1. Detección de Picos
        # 'prominence' asegura que ignoramos micro-fluctuaciones.
        # 'distance' evita detectar el mismo pico dos veces.
        peaks, properties = find_peaks(smoothed_flow, prominence=5, distance=10)
        
        # Si no hay al menos dos picos, no puede haber doble disparo
        if len(peaks) < 2:
            return False, {"reason": "Picos insuficientes (<2)"}
            
        # Analizamos los dos primeros picos principales
        p1_idx = peaks
        p2_idx = peaks[1]
        
        # Verificación temporal: ¿Están los picos temporalmente "apilados"?
        time_diff = time_vector[p2_idx] - time_vector[p1_idx]
        if time_diff > AlgorithmConfig.DT_TIME_WINDOW_SEC:
            return False, {"reason": "Picos demasiado separados", "time_diff": time_diff}
            
        # 2. Análisis del Valle (Morfología)
        # Extraemos el segmento entre el Pico 1 y el Pico 2
        segment = smoothed_flow[p1_idx:p2_idx]
        valley_value = np.min(segment) # El punto más bajo de flujo entre esfuerzos
        
        # 3. Aplicación del Umbral de Valle (Valley Threshold Logic)
        p1_amp = smoothed_flow[p1_idx]
        
        # Definición Matemática:
        # Para respiraciones separadas, el valle debería acercarse a 0 o ser negativo.
        # Para Double Trigger, el valle se mantiene alto ("sin exhalación considerable" ).
        
        is_double_trigger = False
        ratio = 0.0
        
        if p1_amp > 0:
            ratio = valley_value / p1_amp
            
            # SI el flujo en el valle es > 10% del pico (Ratio > 0.10),
            # ENTONCES no hubo relajación suficiente -> Apilamiento Confirmado.
            if ratio > AlgorithmConfig.VALLEY_RETENTION_RATIO:
                is_double_trigger = True
        
        return is_double_trigger, {
            "valley_flow": float(valley_value),
            "peak1_flow": float(p1_amp),
            "valley_retention_ratio": float(ratio),
            "threshold_applied": AlgorithmConfig.VALLEY_RETENTION_RATIO,
            "diagnosis": "Breath Stacking" if is_double_trigger else "Normal Cycling"
        }

@app.route('/health', methods=)
def health_check():
    """Endpoint de verificación de estado del sistema."""
    return jsonify({"status": "Phase 5 Algorithms Operational", "version": "5.0.1"}), 200

@app.route('/analyze/waveform', methods=)
def analyze_waveform():
    """
    Endpoint Principal para Análisis de Precisión (Fase 5).
    Recibe un payload JSON con vectores de tiempo y flujo.
    """
    try:
        data = request.get_json()
        
        # Validación de entrada básica
        if not data or 'time' not in data or 'flow' not in data:
            return jsonify({"error": "Datos de forma de onda incompletos (requiere 'time', 'flow')"}), 400
            
        # Conversión a arrays de Numpy para procesamiento de alta velocidad
        time_vec = np.array(data['time'])
        flow_vec = np.array(data['flow'])
        breath_mode = data.get('mode', 'VC') # Por defecto asume Volumen Control
        
        # --- EJECUCIÓN DE ALGORITMOS DE PRECISIÓN ---
        
        # 1. Análisis de Inanición de Flujo (Área de Déficit)
        deficit_area, ref_area = WaveformProcessor.calculate_area_of_deficit(
            time_vec, flow_vec, breath_mode
        )
        
        # Clasificación de severidad basada en la magnitud del área
        starvation_severity = "None"
        if deficit_area > AlgorithmConfig.DEFICIT_AREA_THRESHOLD:
            starvation_severity = "Severe" if deficit_area > (AlgorithmConfig.DEFICIT_AREA_THRESHOLD * 2) else "Moderate"

        # 2. Análisis de Disparo Doble (Umbral de Valle)
        is_dt, dt_metrics = WaveformProcessor.detect_valley_double_trigger(
            time_vec, flow_vec
        )
        
        # Construcción de la respuesta JSON estructurada
        response = {
            "analysis_metadata": {
                "phase": "Phase 5: Precision Algorithms",
                "algorithm_version": "5.0.1"
            },
            "metrics": {
                "flow_starvation": {
                    "detected": deficit_area > AlgorithmConfig.DEFICIT_AREA_THRESHOLD,
                    "area_of_deficit_value": float(round(deficit_area, 4)),
                    "reference_area": float(round(ref_area, 4)),
                    "deficit_percentage": float(round((deficit_area/ref_area)*100, 2)) if ref_area > 0 else 0,
                    "severity": starvation_severity,
                    "description": "Integral de la diferencia entre flujo ideal y real (Concavidad)"
                },
                "double_trigger": {
                    "detected": is_dt,
                    "valley_metrics": {k: float(round(v, 4)) if isinstance(v, (int, float)) else v for k, v in dt_metrics.items()},
                    "diagnosis": dt_metrics["diagnosis"],
                    "description": "Detección basada en retención de flujo en el valle inspiratorio"
                }
            }
        }
        
        return jsonify(response), 200

    except Exception as e:
        logging.error(f"Error en Procesamiento de Onda: {str(e)}")
        return jsonify({"error": "Error Interno de Procesamiento", "details": str(e)}), 500

if __name__ == '__main__':
    # Configuración de ejecución para desarrollo
    # Para producción, desplegar tras servidor WSGI (Gunicorn/uWSGI)
    app.run(host='0.0.0.0', port=5000, debug=True)
