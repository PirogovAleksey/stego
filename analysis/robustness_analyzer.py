import numpy as np
from scipy import signal
import os


class RobustnessAnalyzer:
    """Клас для тестування стійкості методів стеганографії до різних атак"""

    def __init__(self, audio_reader, msg_processor):
        self.audio_reader = audio_reader
        self.msg_processor = msg_processor
        self.attacks_dir = os.path.join('results', 'attacks')

        # Створюємо каталог для атакованих файлів
        if not os.path.exists(self.attacks_dir):
            os.makedirs(self.attacks_dir)

    def add_noise(self, audio_data, noise_level=0.001):
        """
        Додавання шуму до аудіо

        Args:
            audio_data (numpy.ndarray): Вхідні аудіодані
            noise_level (float): Рівень шуму (0.0-1.0)

        Returns:
            numpy.ndarray: Аудіодані з шумом
        """
        noise = np.random.normal(0, noise_level, len(audio_data))
        return audio_data + (noise * 32767).astype(np.int16)

    def apply_filter(self, audio_data, filter_type='low', cutoff=0.8, order=5):
        """
        Застосування фільтра до аудіо

        Args:
            audio_data (numpy.ndarray): Вхідні аудіодані
            filter_type (str): Тип фільтра ('low', 'high', 'band')
            cutoff (float): Частота зрізу (0.0-1.0)
            order (int): Порядок фільтра

        Returns:
            numpy.ndarray: Відфільтровані аудіодані
        """
        b, a = signal.butter(order, cutoff, filter_type)
        return signal.filtfilt(b, a, audio_data).astype(np.int16)

    def recompress(self, audio_data, quality=32):
        """
        Імітація перекодування/стиснення аудіо

        Args:
            audio_data (numpy.ndarray): Вхідні аудіодані
            quality (int): Параметр якості (більше - краще)

        Returns:
            numpy.ndarray: Перекодовані аудіодані
        """
        return (audio_data / quality).astype(np.int16) * quality

    def cut_audio(self, audio_data, cut_percentage=0.01):
        """
        Обрізання аудіо

        Args:
            audio_data (numpy.ndarray): Вхідні аудіодані
            cut_percentage (float): Відсоток обрізання (0.0-1.0)

        Returns:
            numpy.ndarray: Обрізані аудіодані
        """
        return audio_data[:int(len(audio_data) * (1 - cut_percentage))]

    def scale_amplitude(self, audio_data, scale_factor=0.9):
        """
        Масштабування амплітуди (зміна гучності)

        Args:
            audio_data (numpy.ndarray): Вхідні аудіодані
            scale_factor (float): Коефіцієнт масштабування

        Returns:
            numpy.ndarray: Масштабовані аудіодані
        """
        return np.int16(audio_data * scale_factor)

    def calculate_ber(self, original_message, extracted_message):
        """
        Обчислення Bit Error Rate (BER) для порівняння повідомлень

        Args:
            original_message (str): Оригінальне повідомлення
            extracted_message (str): Витягнуте повідомлення

        Returns:
            float: Bit Error Rate [0.0, 1.0]
        """
        import difflib

        if extracted_message is None or "Не вдалося вилучити повідомлення" in extracted_message:
            return 1.0  # 100% помилок

        # Використання відстані Левенштейна для оцінки подібності
        sequencematcher = difflib.SequenceMatcher(None, original_message, extracted_message)
        similarity = sequencematcher.ratio()
        return 1.0 - similarity

    def test_robustness(self, original_path, stego_path, method_name, extract_function, message):
        """
        Тестування стійкості методу до різних атак

        Args:
            original_path (str): Шлях до оригінального аудіофайлу
            stego_path (str): Шлях до аудіофайлу зі стеганограмою
            method_name (str): Назва методу ('dft', 'dct', 'dwt')
            extract_function (function): Функція для вилучення повідомлення
            message (str): Оригінальне повідомлення для порівняння

        Returns:
            dict: Результати тестування стійкості
        """
        # Зчитування стего-аудіо
        sample_rate, stego_audio = self.audio_reader.read_audio(stego_path)

        # Застосування атак та збереження результатів
        # 1. Атака шумом
        noisy_audio = self.add_noise(stego_audio)
        noise_path = os.path.join(self.attacks_dir, f"attacked_{method_name}_noise.wav")
        self.audio_reader.save_audio(noise_path, sample_rate, noisy_audio)

        # 2. Атака фільтрацією
        filtered_audio = self.apply_filter(stego_audio)
        filter_path = os.path.join(self.attacks_dir, f"attacked_{method_name}_filter.wav")
        self.audio_reader.save_audio(filter_path, sample_rate, filtered_audio)

        # 3. Атака перекодуванням
        recompressed_audio = self.recompress(stego_audio)
        recompress_path = os.path.join(self.attacks_dir, f"attacked_{method_name}_recompress.wav")
        self.audio_reader.save_audio(recompress_path, sample_rate, recompressed_audio)

        # 4. Атака обрізанням
        cut_audio = self.cut_audio(stego_audio)
        cut_path = os.path.join(self.attacks_dir, f"attacked_{method_name}_cut.wav")
        self.audio_reader.save_audio(cut_path, sample_rate, cut_audio)

        # 5. Атака масштабуванням
        scaled_audio = self.scale_amplitude(stego_audio)
        scale_path = os.path.join(self.attacks_dir, f"attacked_{method_name}_scale.wav")
        self.audio_reader.save_audio(scale_path, sample_rate, scaled_audio)

        # Вилучення повідомлень після атак
        results = {}

        # Оригінальне вилучення
        original_extracted = extract_function(stego_path)
        original_success = message == original_extracted
        results["original"] = {
            "success": original_success,
            "extracted": original_extracted,
            "bit_error_rate": 0.0 if original_success else self.calculate_ber(message, original_extracted)
        }

        # Вилучення після атак
        attack_paths = {
            "noise": noise_path,
            "filter": filter_path,
            "recompress": recompress_path,
            "cut": cut_path,
            "scale": scale_path
        }

        for attack_name, attack_path in attack_paths.items():
            try:
                attack_extracted = extract_function(attack_path)
                attack_success = message == attack_extracted
                bit_error_rate = 0.0 if attack_success else self.calculate_ber(message, attack_extracted)
            except Exception as e:
                attack_extracted = f"Помилка вилучення: {str(e)}"
                attack_success = False
                bit_error_rate = 1.0

            results[attack_name] = {
                "success": attack_success,
                "extracted": attack_extracted,
                "bit_error_rate": bit_error_rate
            }

        # Вивід результатів
        print(f"\n====== Результати тестування стійкості методу {method_name.upper()} ======")
        print(f"{'Тип атаки':<15} | {'Успіх':<10} | {'BER (%)':<10} | {'Витягнуте повідомлення'}")
        print("-" * 80)
        for attack_name, attack_result in results.items():
            print(f"{attack_name.capitalize():<15} | {str(attack_result['success']):<10} | "
                  f"{attack_result['bit_error_rate'] * 100:.2f}%{'':<7} | {attack_result['extracted'][:30]}...")

        return results
