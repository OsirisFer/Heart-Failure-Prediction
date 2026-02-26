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
    height_threshold = np.percentile(abs_sig, 98)

    # D) Distancia mínima entre picos (en muestras)
    #    Ej: 0.3 s => no puede haber 2 latidos a menos de 0.3s
    #    (evita detectar múltiples picos dentro del mismo latido)
    min_distance = int(0.3 * fs)

    # E) Encontrar picos
    prom = np.percentile(abs_sig, 95)
    peaks, props = find_peaks(abs_sig, height=height_threshold, distance=min_distance, prominence=prom)

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

def clean_rr(rr, min_s=0.3, max_s=2.0):
    """
    Elimina tiempos entre latidos que son imposibles o claramente errores de detección.
    min_s: menor a esto sería >200 bpm (demasiado rápido)
    max_s: mayor a esto sería <30 bpm (demasiado lento para la mayoría de casos)
    """
    rr = np.array(rr)
    rr = rr[(rr >= min_s) & (rr <= max_s)]
    return rr

def extract_features(rr):
    """
    rr = array de tiempos entre latidos (en segundos)
    Devuelve números resumen (features) para ML.
    """
    rr = np.array(rr)

    return {
        "rr_mean": float(np.mean(rr)),     # promedio del tiempo entre latidos
        "rr_std": float(np.std(rr)),       # variación del ritmo
        "rr_min": float(np.min(rr)),       # latido más rápido (menor tiempo)
        "rr_max": float(np.max(rr)),       # latido más lento (mayor tiempo)
        "rr_count": int(len(rr)),          # cuántos intervalos tenemos (calidad)
    }
    
    

if __name__ == "__main__": #ejecutame si soy el programa principal, no me importes como módulo
    
    X = []  # filas de features
    y = []  # etiquetas: 1=CHF, 0=NSR
    
    # --- CHF ---
    chf = load_ecg_binary(chf_path) # transformamos el archivo binario en un array de numeros para poder usarlo 
    chf_peaks = detect_beats(chf, fs) #cada numero del array representa la amplitud del ECG en cada instante, pero son mediciones todavia
    """
    Como el ECG marca para arriba y para abajo, necesitamos hacer el valor absoluto para tener todos numeros positivos, tambien nos quedamos con 
    los picos grandes (el percentil 99). Porque la mayoria son mediciones hallamos los maximos digamos, los latidos
    Tambien definimos una disntancia minima entre picos para evitar detectar el mismo latido dos veces
    detect_beats entonces nos devuelve posiciones de array, es decir donde ocurrio cada latido, no el valor en si
    plot_with_peaks asegura el funcionamiento
    """
    chf_rr = rr_intervals_seconds(chf_peaks[0], fs)
    chf_rr = clean_rr(chf_rr)
    print("CHF RR count after clean:", len(chf_rr))
    if len(chf_rr) > 0:
        print("CHF RR mean/std after clean:", float(np.mean(chf_rr)), float(np.std(chf_rr)))

    chf_feats = extract_features(chf_rr)
    print("CHF features:", chf_feats)

    X.append([chf_feats["rr_mean"], chf_feats["rr_std"], chf_feats["rr_min"], chf_feats["rr_max"]])
    y.append(1)

    """
    ahora con los indices de los latidos, pasamos de ECG a ritmo cardiaco, usamos el valor de los indices y los convertimos a tiempo real, 
    como veniamos usadno 1/128 segundos dividimos sobre 128 para volver a segundos entonces 
    peaks = [512, 640, 768]
    dividido por 128 da:
    beat_times = [4.0, 5.0, 6.0] segundos
    esto nos dice que hubo latidos en 4 5 y 6, aplico np.diff [5.0 - 4.0, 6.0 - 5.0] → [1.0, 1.0]
    eso nos da el tiempo entre latidos
    """
    print("CHF: picos detectados:", len(chf_peaks[0]))

    print("CHF height threshold:", chf_peaks[2])
    print("CHF total picos reales:", len(chf_peaks[0]))

    if len(chf_rr) > 0:
        print("CHF: RR promedio (s):", float(np.mean(chf_rr)))
        print("CHF: RR variación (std):", float(np.std(chf_rr)))

    plot_with_peaks(chf, fs, chf_peaks, "CHF chf201 - 10s con picos")

    # --- NSR ---
    nsr = load_ecg_binary(nsr_path)
    nsr_peaks, nsr_abs, nsr_thr = detect_beats(nsr, fs)
    nsr_rr = rr_intervals_seconds(nsr_peaks, fs)
    nsr_rr = clean_rr(nsr_rr)
    print("NSR RR count after clean:", len(nsr_rr))
    if len(nsr_rr) > 0:
        print("NSR RR mean/std after clean:", float(np.mean(nsr_rr)), float(np.std(nsr_rr)))
    

    nsr_feats = extract_features(nsr_rr)
    print("NSR features:", nsr_feats)

    X.append([nsr_feats["rr_mean"], nsr_feats["rr_std"], nsr_feats["rr_min"], nsr_feats["rr_max"]])
    y.append(0)

    print("NSR: picos detectados:", len(nsr_peaks))

    print("NSR height threshold:", nsr_thr)
    print("NSR total picos reales:", len(nsr_peaks))

    if len(nsr_rr) > 0:
        print("NSR: RR promedio (s):", float(np.mean(nsr_rr)))
        print("NSR: RR variación (std):", float(np.std(nsr_rr)))

    plot_with_peaks(nsr, fs, nsr_peaks, "NSR nsr001 - 10s con picos")

    X = np.array(X)
    y = np.array(y)

    print("X:", X)
    print("y:", y)

    