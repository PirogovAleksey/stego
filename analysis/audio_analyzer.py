import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os


class AudioAnalyzer:
    """Клас для аналізу якості аудіо та порівняння оригінального і модифікованого"""

    def __init__(self, audio_reader):
        self.audio_reader = audio_reader
        self.analysis_dir = os.path.join('results', 'analysis')

        # Створюємо каталог для результатів аналізу
        if not os.path.exists(self.analysis_dir):
            os.makedirs(self.analysis_dir)

    def compare_audio(self, original, modified, title="audio_comparison"):
        """
        Порівняння якості аудіо до та після вбудовування

        Args:
            original (numpy.ndarray): Оригінальні аудіодані
            modified (numpy.ndarray): Модифіковані аудіодані
            title (str): Назва для графіку

        Returns:
            float: SNR (Signal-to-Noise Ratio)
        """
        # Обчислення SNR (Signal-to-Noise Ratio)
        noise = original - modified
        snr = 10 * np.log10(np.sum(original ** 2) / np.sum(noise ** 2))

        # Візуалізація аудіосигналів
        plt.figure(figsize=(12, 8))

        # Оригінальний сигнал
        plt.subplot(3, 1, 1)
        plt.plot(original[:1000])
        plt.title('Оригінальний аудіосигнал (перші 1000 відліків)')
        plt.xlabel('Відлік')
        plt.ylabel('Амплітуда')

        # Модифікований сигнал
        plt.subplot(3, 1, 2)
        plt.plot(modified[:1000])
        plt.title('Модифікований аудіосигнал (перші 1000 відліків)')
        plt.xlabel('Відлік')
        plt.ylabel('Амплітуда')

        # Різниця (шум)
        plt.subplot(3, 1, 3)
        plt.plot(noise[:1000])
        plt.title(f'Різниця (шум), SNR = {snr:.2f} дБ')
        plt.xlabel('Відлік')
        plt.ylabel('Амплітуда')

        plt.tight_layout()
        plt.savefig(os.path.join(self.analysis_dir, f'{title}.png'))
        plt.close()

        return snr

    def spectrogram_comparison(self, original_path, modified_path, title="spectrogram_comparison"):
        """
        Порівняння спектрограм оригінального та модифікованого аудіо

        Args:
            original_path (str): Шлях до оригінального аудіофайлу
            modified_path (str): Шлях до модифікованого аудіофайлу
            title (str): Назва для графіку
        """
        # Зчитування аудіофайлів
        sr1, original = self.audio_reader.read_audio(original_path)
        sr2, modified = self.audio_reader.read_audio(modified_path)

        # Обчислення та візуалізація спектрограм
        plt.figure(figsize=(12, 8))

        # Спектрограма оригінального аудіо
        plt.subplot(2, 1, 1)
        plt.specgram(original, Fs=sr1, NFFT=1024, noverlap=512)
        plt.title('Спектрограма оригінального аудіо')
        plt.xlabel('Час (с)')
        plt.ylabel('Частота (Гц)')
        plt.colorbar(format='%+2.0f dB')

        # Спектрограма модифікованого аудіо
        plt.subplot(2, 1, 2)
        plt.specgram(modified, Fs=sr2, NFFT=1024, noverlap=512)
        plt.title('Спектрограма модифікованого аудіо')
        plt.xlabel('Час (с)')
        plt.ylabel('Частота (Гц)')
        plt.colorbar(format='%+2.0f dB')

        plt.tight_layout()
        plt.savefig(os.path.join(self.analysis_dir, f'{title}.png'))
        plt.close()

    def analyze_audio_quality(self, original_path, stego_path, title="audio_quality_analysis"):
        """
        Комплексний аналіз якості аудіо

        Args:
            original_path (str): Шлях до оригінального аудіофайлу
            stego_path (str): Шлях до аудіофайлу зі стеганограмою
            title (str): Назва для графіку

        Returns:
            dict: Словник з метриками якості
        """
        # Завантаження аудіо з використанням librosa
        y_original, sr_original = librosa.load(original_path, sr=None)
        y_stego, sr_stego = librosa.load(stego_path, sr=None)

        # Переконуємося, що довжини аудіо однакові
        min_len = min(len(y_original), len(y_stego))
        y_original = y_original[:min_len]
        y_stego = y_stego[:min_len]

        # 1. Обчислення SNR
        noise = y_original - y_stego
        snr = 10 * np.log10(np.sum(y_original ** 2) / np.sum(noise ** 2))

        # 2. Обчислення PSNR (Peak Signal-to-Noise Ratio)
        max_possible_amplitude = np.max(np.abs(y_original))
        psnr = 20 * np.log10(max_possible_amplitude / np.sqrt(np.mean(noise ** 2)))

        # 3. Обчислення MSE (Mean Squared Error)
        mse = np.mean((y_original - y_stego) ** 2)

        # 4. Обчислення гістограм та MFCC
        plt.figure(figsize=(12, 10))

        plt.subplot(2, 2, 1)
        plt.hist(y_original, bins=100, alpha=0.7, label='Оригінал')
        plt.hist(y_stego, bins=100, alpha=0.7, label='Стего')
        plt.title('Гістограми амплітуд')
        plt.legend()

        # 5. Обчислення MFCC (Mel-frequency cepstral coefficients)
        mfcc_original = librosa.feature.mfcc(y=y_original, sr=sr_original, n_mfcc=13)
        mfcc_stego = librosa.feature.mfcc(y=y_stego, sr=sr_stego, n_mfcc=13)

        plt.subplot(2, 2, 2)
        librosa.display.specshow(mfcc_original, x_axis='time')
        plt.title('MFCC Оригінал')
        plt.colorbar()

        plt.subplot(2, 2, 3)
        librosa.display.specshow(mfcc_stego, x_axis='time')
        plt.title('MFCC Стего')
        plt.colorbar()

        # 6. Обчислення різниці MFCC
        mfcc_diff = np.abs(mfcc_original - mfcc_stego)

        plt.subplot(2, 2, 4)
        librosa.display.specshow(mfcc_diff, x_axis='time')
        plt.title('Різниця MFCC')
        plt.colorbar()

        plt.tight_layout()
        plt.savefig(os.path.join(self.analysis_dir, f'{title}.png'))
        plt.close()

        # Вивід результатів
        print(f"SNR: {snr:.2f} дБ")
        print(f"PSNR: {psnr:.2f} дБ")
        print(f"MSE: {mse:.8f}")

        return {
            'SNR': snr,
            'PSNR': psnr,
            'MSE': mse
        }
