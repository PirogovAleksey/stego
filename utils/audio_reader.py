import numpy as np
from scipy.io import wavfile
import warnings
import os

class AudioReader:
    """Клас для роботи з аудіофайлами - зчитування, конвертування і збереження"""

    @staticmethod
    def read_audio(audio_path):
        """
        Зчитує аудіофайл у форматі float32

        Args:
            audio_path (str): Шлях до аудіофайлу

        Returns:
            tuple: (частота дискретизації, аудіодані)
        """
        # Ігноруємо попередження для WAV файлів
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", wavfile.WavFileWarning)
            sample_rate, audio_data = wavfile.read(audio_path)

        # Конвертуємо у float32 з нормалізацією
        if audio_data.dtype == np.int16:
            print(f"[DEBUG] Конвертація з int16 у float32")
            # Нормалізуємо до [-1.0, 1.0]
            audio_data = audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype != np.float32:
            print(f"[DEBUG] Конвертація з {audio_data.dtype} у float32")
            audio_data = audio_data.astype(np.float32)

        print(f"[DEBUG] Зчитано аудіо: shape={audio_data.shape}, dtype={audio_data.dtype}, sr={sample_rate}")

        return sample_rate, audio_data

    @staticmethod
    def save_audio(audio_path, sample_rate, audio_data):
        """
        Зберігає аудіодані у файл у форматі float32

        Args:
            audio_path (str): Шлях для збереження
            sample_rate (int): Частота дискретизації
            audio_data (numpy.ndarray): Аудіодані
        """
        # Конвертуємо у float32 та нормалізуємо, якщо потрібно
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)

        # Переконуємося, що значення в розумних межах для float32 WAV (-1.0 до 1.0)
        if np.max(np.abs(audio_data)) > 1.0:
            print("[DEBUG] Нормалізація аудіо до діапазону [-1.0, 1.0]")
            max_val = np.max(np.abs(audio_data))
            audio_data = audio_data / max_val

        # Переконуємося, що директорія існує
        directory = os.path.dirname(audio_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        # Збереження у форматі float32
        wavfile.write(audio_path, sample_rate, audio_data)
        print(f"[DEBUG] Збережено аудіо у {audio_path}: shape={audio_data.shape}, dtype={audio_data.dtype}")
