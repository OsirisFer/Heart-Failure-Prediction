import os  # rutas a archivos
import numpy as np  # arrays y operaciones numéricas
import matplotlib.pyplot as plt  # gráficos
from scipy.signal import find_peaks  # detector de picos en señales

# 1) Rutas a los archivos ECG
chf_path = os.path.join("pacientes con ataque", "chf201.ecg")
nsr_path = os.path.join("pacientes sin ataque", "nsr001.ecg")

"""
os es un módulo de Python para interactuar con el sistema operativo. 
path es un submódulo de os que tiene funciones para manejar rutas de archivos.
join es una función que toma partes de una ruta y las une de forma correcta para el sistema

"""

# 2) fs: 128 muestras/segundo (sale de tu .hea)
fs = 128

def load_ecg_binary(path):
    """
    Lee un archivo binario .ecg interpretándolo como int16.
    """
    return np.fromfile(path, dtype=np.int16)


def detect_beats(signal, fs):
    """
    Detecta picos (latidos) con reglas simples.
    Devuelve:
    - peaks: índices (posiciones) en el array donde hay picos
    """
    # A) Centramos la señal: le quitamos el "nivel base" (promedio)
    #    Esto ayuda porque algunos ECG vienen "corridos" hacia arriba.
    centered = signal - np.mean(signal)

    # B) Tomamos valor absoluto por si los picos principales fueran negativos
    #    (algunos ECG tienen el latido como valle en vez de pico)
    abs_sig = np.abs(centered)

    # C) Umbral automático: "picos que sean bastante grandes"
    #    percentil 99 = tomamos el 1% más alto como candidato a pico
    height_threshold = np.percentile(abs_sig, 99)

    # D) Distancia mínima entre picos (en muestras)
    #    Ej: 0.3 s => no puede haber 2 latidos a menos de 0.3s
    #    (evita detectar múltiples picos dentro del mismo latido)
    min_distance = int(0.3 * fs)

    # E) Encontrar picos
    peaks, props = find_peaks(abs_sig, height=height_threshold, distance=min_distance)

    return peaks, abs_sig, height_threshold

def plot_with_peaks(signal, fs, peaks, title):
    """
    Grafica los primeros 'seconds' segundos y marca los picos detectados.
    """
    if isinstance(peaks, tuple):
        peaks = peaks[0]
    n = int(fs * 10)
    y = signal[:n]
    x = np.arange(len(y)) / fs

    # Nos quedamos con picos que caen dentro de esos primeros n puntos
    peaks_in_window = peaks[peaks < n]

    plt.figure()
    plt.plot(x, y)
    plt.plot(peaks_in_window / fs, y[peaks_in_window], "x")  # marca picos con X
    plt.title(title)
    plt.xlabel("Tiempo (s)")
    plt.ylabel("ECG (crudo)")
    plt.show()

def rr_intervals_seconds(peaks, fs):
    """
    Convierte índices de picos a tiempos entre latidos (en segundos).
    """
    # Convertimos posiciones (muestras) a segundos
    beat_times = peaks / fs

    # Diferencias entre latido i y latido i+1
    rr = np.diff(beat_times)

    return rr
    

if __name__ == "__main__": #ejecutame si soy el programa principal, no me importes como módulo
    # --- CHF ---
    chf = load_ecg_binary(chf_path) #carga el archivo binario .ecg como un array de int16
    chf_peaks = detect_beats(chf, fs) 
    chf_rr = rr_intervals_seconds(chf_peaks[0], fs)

    print("CHF: picos detectados:", len(chf_peaks))
    if len(chf_rr) > 0:
        print("CHF: RR promedio (s):", float(np.mean(chf_rr)))
        print("CHF: RR variación (std):", float(np.std(chf_rr)))

    plot_with_peaks(chf, fs, chf_peaks, "CHF chf201 - 10s con picos")

    # --- NSR ---
    nsr = load_ecg_binary(nsr_path)
    nsr_peaks, nsr_abs, nsr_thr = detect_beats(nsr, fs)
    nsr_rr = rr_intervals_seconds(nsr_peaks, fs)

    print("NSR: picos detectados:", len(nsr_peaks))
    if len(nsr_rr) > 0:
        print("NSR: RR promedio (s):", float(np.mean(nsr_rr)))
        print("NSR: RR variación (std):", float(np.std(nsr_rr)))

    plot_with_peaks(nsr, fs, nsr_peaks, "NSR nsr001 - 10s con picos")