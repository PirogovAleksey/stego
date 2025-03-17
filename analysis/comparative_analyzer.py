import numpy as np
import matplotlib.pyplot as plt
import os
import time


class ComparativeAnalyzer:
    """Клас для порівняльного аналізу різних методів стеганографії"""

    def __init__(self, audio_analyzer, robustness_analyzer):
        """
        Ініціалізація класу аналізатора порівнянь

        Args:
            audio_analyzer: Об'єкт аналізатора аудіо
            robustness_analyzer: Об'єкт аналізатора стійкості
        """
        self.audio_analyzer = audio_analyzer
        self.robustness_analyzer = robustness_analyzer
        self.analysis_dir = os.path.join('results', 'analysis')

        # Створюємо каталог для результатів
        if not os.path.exists(self.analysis_dir):
            os.makedirs(self.analysis_dir)

    def compare_methods(self, audio_path, message, methods_dict, output_prefix=None):
        """
        Порівняння різних методів стеганографії за різними критеріями

        Args:
            audio_path (str): Шлях до оригінального аудіофайлу
            message (str): Повідомлення для вбудовування
            methods_dict (dict): Словник {назва_методу: (об'єкт_методу, функція_вилучення)}
            output_prefix (str, optional): Префікс для вихідних файлів

        Returns:
            dict: Результати порівняння
        """
        if output_prefix is None:
            output_prefix = os.path.join('results', 'steganography',
                                         os.path.splitext(os.path.basename(audio_path))[0])

        # Створюємо каталог для вихідних файлів
        output_dir = os.path.dirname(output_prefix)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        results = {}

        # Тестування кожного методу
        for method_name, (method_obj, extract_func) in methods_dict.items():
            print(f"\n===== Тестування методу {method_name.upper()} =====")

            # Вимірювання часу вбудовування
            start_time = time.time()
            output_path = f"{output_prefix}_{method_name.lower()}.wav"
            _, _ = method_obj.embed(audio_path, message, output_path)
            embedding_time = time.time() - start_time

            # Вимірювання часу вилучення
            start_time = time.time()
            extracted_message = extract_func(output_path)
            extraction_time = time.time() - start_time

            # Перевірка успішності
            success = message == extracted_message

            # Аналіз якості
            quality_results = self.audio_analyzer.analyze_audio_quality(
                audio_path, output_path, f"{method_name.lower()}_quality")

            # Аналіз стійкості
            robustness_results = self.robustness_analyzer.test_robustness(
                audio_path, output_path, method_name.lower(), extract_func, message)

            # Формування результатів
            results[method_name] = {
                'embedding_time': embedding_time,
                'extraction_time': extraction_time,
                'success': success,
                'bit_error_rate': 0.0 if success else self.robustness_analyzer.calculate_ber(
                    message, extracted_message),
                'quality': quality_results,
                'robustness': robustness_results
            }

        # Вивід порівняльної таблиці
        method_names = list(results.keys())
        print("\n====== Порівняння методів ======")
        headers = ['Характеристика'] + method_names
        row_format = "{:<20} | " + " | ".join(["{:<15}" for _ in method_names])
        print(row_format.format(*headers))
        print("-" * (22 + 17 * len(method_names)))

        # Виведення характеристик
        metrics = [
            ('Час вбудовування (с)', lambda m: results[m]['embedding_time']),
            ('Час вилучення (с)', lambda m: results[m]['extraction_time']),
            ('Успішність вилучення', lambda m: str(results[m]['success'])),
            ('SNR (дБ)', lambda m: results[m]['quality']['SNR']),
            ('PSNR (дБ)', lambda m: results[m]['quality']['PSNR']),
            ('MSE', lambda m: results[m]['quality']['MSE'])
        ]

        for metric_name, metric_func in metrics:
            values = [metric_func(m) for m in method_names]
            # Форматування числових значень
            formatted_values = []
            for v in values:
                if isinstance(v, (int, float)):
                    formatted_values.append(f"{v:.5f}" if v < 0.1 else f"{v:.2f}")
                else:
                    formatted_values.append(v)
            print(row_format.format(metric_name, *formatted_values))

        # Створення зведених графіків
        self.plot_quality_comparison(results)
        self.plot_robustness_comparison(results)

        return results

    def plot_quality_comparison(self, results):
        """
        Побудова графіків порівняння якості для різних методів

        Args:
            results (dict): Результати порівняння методів
        """
        methods = list(results.keys())
        snr_values = [results[method]['quality']['SNR'] for method in methods]
        psnr_values = [results[method]['quality']['PSNR'] for method in methods]
        mse_values = [results[method]['quality']['MSE'] for method in methods]

        plt.figure(figsize=(15, 10))

        # SNR
        plt.subplot(2, 2, 1)
        plt.bar(methods, snr_values, color=['blue', 'green', 'red'])
        plt.title('Порівняння SNR')
        plt.ylabel('SNR (дБ)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # PSNR
        plt.subplot(2, 2, 2)
        plt.bar(methods, psnr_values, color=['blue', 'green', 'red'])
        plt.title('Порівняння PSNR')
        plt.ylabel('PSNR (дБ)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # MSE
        plt.subplot(2, 2, 3)
        plt.bar(methods, mse_values, color=['blue', 'green', 'red'])
        plt.title('Порівняння MSE')
        plt.ylabel('MSE')
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Час виконання
        plt.subplot(2, 2, 4)
        embed_times = [results[method]['embedding_time'] for method in methods]
        extract_times = [results[method]['extraction_time'] for method in methods]

        x = np.arange(len(methods))
        width = 0.35

        plt.bar(x - width / 2, embed_times, width, label='Час вбудовування')
        plt.bar(x + width / 2, extract_times, width, label='Час вилучення')

        plt.xticks(x, methods)
        plt.ylabel('Час (с)')
        plt.title('Порівняння часу виконання')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig(os.path.join(self.analysis_dir, 'quality_comparison.png'))
        plt.close()

    def plot_robustness_comparison(self, results):
        """
        Побудова графіків порівняння стійкості для різних методів

        Args:
            results (dict): Результати порівняння методів
        """
        methods = list(results.keys())
        attacks = ['noise', 'filter', 'recompress', 'cut', 'scale']
        attack_names = ['Шум', 'Фільтрація', 'Перекодування', 'Обрізання', 'Масштабування']

        # Підготовка даних для графіка
        success_data = []
        ber_data = []

        for method in methods:
            method_success = []
            method_ber = []

            for attack in attacks:
                attack_results = results[method]['robustness'][attack]
                method_success.append(1 if attack_results['success'] else 0)
                method_ber.append(attack_results['bit_error_rate'])

            success_data.append(method_success)
            ber_data.append(method_ber)

        plt.figure(figsize=(15, 10))

        # Графік успішності вилучення
        plt.subplot(2, 1, 1)

        x = np.arange(len(attack_names))
        width = 0.25

        for i, method in enumerate(methods):
            plt.bar(x + (i - 1) * width, success_data[i], width, label=method)

        plt.xticks(x, attack_names)
        plt.ylabel('Успішність (1 - так, 0 - ні)')
        plt.title('Порівняння стійкості до атак (успішність вилучення)')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Графік BER
        plt.subplot(2, 1, 2)

        for i, method in enumerate(methods):
            plt.bar(x + (i - 1) * width, ber_data[i], width, label=method)

        plt.xticks(x, attack_names)
        plt.ylabel('Bit Error Rate')
        plt.title('Порівняння стійкості до атак (BER)')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig(os.path.join(self.analysis_dir, 'robustness_comparison.png'))
        plt.close()

    def analyze_capacity(self, audio_path, methods_dict, message_sizes=[50, 100, 200, 500, 1000]):
        """
        Аналіз впливу розміру повідомлення на якість аудіо

        Args:
            audio_path (str): Шлях до оригінального аудіофайлу
            methods_dict (dict): Словник {назва_методу: (об'єкт_методу, функція_вилучення)}
            message_sizes (list): Список розмірів повідомлень для тестування

        Returns:
            dict: Результати аналізу
        """
        results = {method_name: {'snr': [], 'psnr': [], 'mse': [], 'success': []}
                   for method_name in methods_dict.keys()}

        # Створюємо базове повідомлення
        base_message = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "

        for size in message_sizes:
            # Створення повідомлення необхідної довжини
            message = base_message * (size // len(base_message) + 1)
            message = message[:size]

            print(f"\nТестування повідомлення довжиною {size} символів...")

            for method_name, (method_obj, extract_func) in methods_dict.items():
                try:
                    # Вбудовування
                    output_path = os.path.join('results', 'steganography',
                                               f"capacity_{method_name.lower()}_{size}.wav")
                    method_obj.embed(audio_path, message, output_path)

                    # Аналіз якості
                    quality = self.audio_analyzer.analyze_audio_quality(
                        audio_path, output_path, f"capacity_{method_name.lower()}_{size}")
                    results[method_name]['snr'].append(quality['SNR'])
                    results[method_name]['psnr'].append(quality['PSNR'])
                    results[method_name]['mse'].append(quality['MSE'])

                    # Перевірка успішності
                    extracted = extract_func(output_path)
                    success = message == extracted
                    results[method_name]['success'].append(1 if success else 0)

                    print(f"{method_name}: SNR = {quality['SNR']:.2f} дБ, "
                          f"PSNR = {quality['PSNR']:.2f} дБ, Успіх = {success}")

                except Exception as e:
                    print(f"Помилка при тестуванні методу {method_name} з розміром {size}: {str(e)}")
                    results[method_name]['snr'].append(None)
                    results[method_name]['psnr'].append(None)
                    results[method_name]['mse'].append(None)
                    results[method_name]['success'].append(0)

        # Побудова графіків
        plt.figure(figsize=(15, 10))

        # SNR
        plt.subplot(2, 2, 1)
        for method_name in methods_dict.keys():
            plt.plot(message_sizes, results[method_name]['snr'], marker='o', label=method_name)
        plt.title('Залежність SNR від розміру повідомлення')
        plt.xlabel('Розмір повідомлення (символів)')
        plt.ylabel('SNR (дБ)')
        plt.legend()
        plt.grid(True)

        # PSNR
        plt.subplot(2, 2, 2)
        for method_name in methods_dict.keys():
            plt.plot(message_sizes, results[method_name]['psnr'], marker='o', label=method_name)
        plt.title('Залежність PSNR від розміру повідомлення')
        plt.xlabel('Розмір повідомлення (символів)')
        plt.ylabel('PSNR (дБ)')
        plt.legend()
        plt.grid(True)

        # MSE
        plt.subplot(2, 2, 3)
        for method_name in methods_dict.keys():
            plt.plot(message_sizes, results[method_name]['mse'], marker='o', label=method_name)
        plt.title('Залежність MSE від розміру повідомлення')
        plt.xlabel('Розмір повідомлення (символів)')
        plt.ylabel('MSE')
        plt.legend()
        plt.grid(True)

        # Успішність
        plt.subplot(2, 2, 4)
        for method_name in methods_dict.keys():
            plt.plot(message_sizes, results[method_name]['success'], marker='o', label=method_name)
        plt.title('Успішність вилучення залежно від розміру повідомлення')
        plt.xlabel('Розмір повідомлення (символів)')
        plt.ylabel('Успішність (1 - так, 0 - ні)')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.analysis_dir, 'capacity_analysis.png'))
        plt.close()

        return results
