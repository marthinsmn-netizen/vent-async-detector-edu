# 🫁 Ventilator Lab: Detector Híbrido de Asincronías

> **Sistema de Apoyo a la Decisión Clínica (CDSS) basado en Visión Artificial y GenAI.** > *Herramienta educativa avanzada para la identificación de asincronías paciente-ventilador en tiempo real.*

![Estado](https://img.shields.io/badge/Estado-Prototipo_Funcional-blue?style=flat-square)
![Stack](https://img.shields.io/badge/Tech-Python_|_Streamlit_|_Gemini_1.5-green?style=flat-square)
![Core](https://img.shields.io/badge/IP-Visión_Híbrida-orange?style=flat-square)
![Licencia](https://img.shields.io/badge/Licencia-Propiedad_Intelectual_Privada-red?style=flat-square)

---

## 📋 Resumen Ejecutivo

**Ventilator Lab** es una solución de software diseñada para asistir a médicos intensivistas, terapeutas respiratorios y estudiantes en la interpretación de curvas de ventilación mecánica.

El sistema resuelve el problema de la variabilidad en la interpretación humana mediante una **Arquitectura Híbrida de Doble Validación**:
1.  **Capa Determinista (OpenCV + SciPy):** Realiza un análisis geométrico instantáneo de la señal para detectar anomalías matemáticas basándose en reglas heurísticas predefinidas.
2.  **Capa Generativa (Google Gemini 1.5):** Actúa como un "consultor experto" (Second Opinion), analizando la morfología visual completa y el contexto clínico para validar el hallazgo y reducir falsos positivos.

---

## 🚀 Características Clave (IP Core)

### 1. Extracción de Señal "Hardware-Agnostic"
El sistema no requiere integración física con el ventilador (cables RS232/HL7). Utiliza **Visión por Computadora** para digitalizar las curvas directamente desde la pantalla, haciéndolo universalmente compatible con cualquier marca (Hamilton, Dräger, Maquet, Puritan Bennett, etc.).

### 2. Motor de Calibración Dinámica (HSV)
Incluye una interfaz de calibración en tiempo real que permite al usuario ajustar los filtros de color (Matiz, Saturación, Brillo) para aislar la curva de interés, eliminando ruido causado por reflejos o condiciones de luz variables en la UCI.

### 3. Detección Clínica Específica
El algoritmo híbrido identifica patrones complejos:
* **Doble Disparo (Double Trigger):** Detección por proximidad temporal de picos (<1.0s) y análisis de profundidad del valle exhalatorio.
* **Hambre de Flujo (Flow Starvation):** Detección de concavidades anómalas (muescas) en la rama inspiratoria de la curva de presión-tiempo.
* **Análisis Contextual:** La capa de IA evalúa la morfología global para descartar artefactos.

---

## 🛠️ Arquitectura Técnica

El siguiente diagrama ilustra el flujo de datos desde la captura hasta el diagnóstico:

```mermaid
graph TD
    A[📸 Cámara / Imagen Input] --> B{Pre-Procesamiento OpenCV}
    B -->|Conversión HSV| C[Máscara de Color Adaptativa]
    C -->|Extracción Coordenadas| D[Señal Cruda 1D]
    D -->|Filtro Savitzky-Golay| E[Señal Suavizada]
    
    E --> F{⚙️ MOTOR 1: Matemático}
    F -->|SciPy find_peaks| G[Análisis de Geometría]
    G --> H[Reglas Heurísticas]
    H -->|Output Rápido| I[Diagnóstico Preliminar]
    
    A --> J{🧠 MOTOR 2: GenAI}
    J -->|API Request| K[Google Gemini 1.5 Pro/Flash]
    K -->|Prompt Engineering: Rol Médico| L[Análisis Morfológico]
    
    I --> M[🖥️ Interfaz de Usuario]
    L --> M
💻 Instalación y Uso
Requisitos Previos
Python 3.9 o superior.

Una API Key de Google AI Studio (para la funcionalidad de IA).

1. Clonar el Repositorio
git clone [https://github.com/tu-usuario/vent-async-detector-edu.git](https://github.com/tu-usuario/vent-async-detector-edu.git)
cd vent-async-detector-edu
2. Instalar Dependencias
pip install -r requirements.txt

3. Configuración de API Key (Seguridad)
El sistema gestiona las credenciales de forma segura para despliegues públicos:

Modo Producción: Configurar el "Secret" GOOGLE_API_KEY en el panel de Streamlit Cloud.

Modo Usuario: El usuario puede ingresar su propia clave temporalmente en la barra lateral de la aplicación.

4. Ejecutar la Aplicación
streamlit run app.py

🏥 Validación y Seguridad
El software implementa un mecanismo de "Autodescubrimiento de Modelos". Si el modelo de IA preferido (gemini-1.5-flash) no está disponible en la región del usuario, el sistema iterará automáticamente por una lista de modelos compatibles (pro, vision, latest) hasta lograr la conexión, garantizando una alta disponibilidad.

⚠️ Aviso Legal y Descargo de Responsabilidad (Disclaimer)
LEA ATENTAMENTE ANTES DE USAR:

Herramienta Educativa: Este software es una prueba de concepto y una herramienta de soporte educativo. NO es un dispositivo médico certificado (FDA, CE, ANMAT, etc.).

Responsabilidad: El software no sustituye el juicio clínico profesional. No debe utilizarse como única base para tomar decisiones críticas sobre el soporte vital o la medicación del paciente.

Privacidad: El análisis de imágenes se realiza en la nube (para la función de IA). Asegúrese de no capturar datos identificables del paciente (nombre, historia clínica) en las fotografías.

📞 Contacto y Propiedad Intelectual
Este proyecto representa una Propiedad Intelectual (IP) activa. Para consultas sobre licencias comerciales, colaboración académica o acceso al "White Paper" completo:

Desarrollador Principal: Juan Martín Nuñez Silveira

Email: juanm.nunez@hospitalitaliano.org.ar

LinkedIn: (https://www.linkedin.com/in/juan-mart%C3%ADn-nu%C3%B1ez-silveira-07452058/)

Developed with ❤️ for Critical Care Medicine.
