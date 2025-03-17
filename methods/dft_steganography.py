import numpy as np
from scipy.fft import fft, ifft


class DFTSteganographyFloat32:
    """Клас для стеганографії з використанням DFT з підтримкою float32"""

    END_MARKER = '1111111111111110'  # Маркер кінця повідомлення

    def __init__(self, audio_reader, msg_processor):
        self.audio_reader = audio_reader
        self.msg_processor = msg_processor

        # Налаштування (можна змінювати перед запуском)
        self.start_idx = 2000
        self.step = 2
        self.bit1_value = 0.01
        self.bit0_value = 0.001
        self.threshold = 0.005

    def embed(self, audio_path, message, output_path):
        sample_rate, audio_data = self.audio_reader.read_audio(audio_path)

        if audio_data.ndim == 2:
            print("[DEBUG] Стерео виявлено, працюємо з першим каналом")
            channel_1 = audio_data[:, 0].astype(np.float32)
        else:
            channel_1 = audio_data.astype(np.float32)

        # Нормалізація
        channel_1 /= np.max(np.abs(channel_1))

        # Підготовка повідомлення
        binary_message = self.msg_processor.text_to_binary(message) + self.END_MARKER
        print(f"[DEBUG] Бінарне повідомлення ({len(binary_message)} біт): {binary_message[:50]}...")

        # Перевірка доступного місця
        max_bits = (len(channel_1) // 2 - self.start_idx) // self.step
        if len(binary_message) > max_bits:
            raise ValueError(f"Повідомлення занадто велике. Максимум {max_bits} біт.")

        # DFT
        channel_dft = fft(channel_1)
        max_amp = np.max(np.abs(channel_dft))
        print(f"[DEBUG] Максимальна амплітуда в спектрі: {max_amp}")

        # Вбудовування
        for i, bit in enumerate(binary_message):
            idx = self.start_idx + i * self.step
            if idx >= len(channel_dft) // 2:
                break

            phase = np.angle(channel_dft[idx])

            amp = self.bit1_value if bit == '1' else self.bit0_value
            amp = min(amp, max_amp * 0.1)  # Не перевищувати 10% від макс амплітуди

            channel_dft[idx] = amp * np.exp(1j * phase)
            channel_dft[-idx] = np.conj(channel_dft[idx])  # Симетрія

        print(f"[DEBUG] Вбудовано {len(binary_message)} біт")

        # Інверсний DFT
        modified_channel = np.real(ifft(channel_dft)).astype(np.float32)

        # Нормалізація після зворотного перетворення
        max_val = np.max(np.abs(modified_channel))
        if max_val > 1.0:
            modified_channel /= max_val
            print(f"[DEBUG] Нормалізація виконана, новий максимум: {np.max(np.abs(modified_channel))}")

        # Запис результату
        if audio_data.ndim == 2:
            modified_audio = audio_data.copy()
            modified_audio[:, 0] = modified_channel
        else:
            modified_audio = modified_channel

        self.audio_reader.save_audio(output_path, sample_rate, modified_audio)
        print(f"[DEBUG] Повідомлення вбудовано та збережено в {output_path}")

        return audio_data, modified_audio

    def extract(self, audio_path):
        sample_rate, audio_data = self.audio_reader.read_audio(audio_path)

        if audio_data.ndim == 2:
            print("[DEBUG] Стерео виявлено, працюємо з першим каналом")
            channel_1 = audio_data[:, 0].astype(np.float32)
        else:
            channel_1 = audio_data.astype(np.float32)

        # DFT
        channel_dft = fft(channel_1)

        max_bits = (len(channel_1) // 2 - self.start_idx) // self.step
        binary_message = ""

        print(f"[DEBUG] Поріг для розпізнавання бітів: {self.threshold}")

        # Для дебагу: показати перші 10 амплітуд
        print("[DEBUG] Амплітуди перших 10 коефіцієнтів:")
        for i in range(10):
            idx = self.start_idx + i * self.step
            amp = abs(channel_dft[idx])
            print(f"Коефіцієнт {idx}: {amp:.8f} -> {'1' if amp > self.threshold else '0'}")

        # Витягування бітів
        for i in range(max_bits):
            idx = self.start_idx + i * self.step
            if idx >= len(channel_dft) // 2:
                break

            amplitude = abs(channel_dft[idx])
            bit = '1' if amplitude > self.threshold else '0'
            binary_message += bit

            # Перевірка маркера кінця
            if binary_message.endswith(self.END_MARKER):
                binary_message = binary_message[:-len(self.END_MARKER)]
                print(f"[DEBUG] Маркер кінця знайдено на позиції {i}")
                break

        if not binary_message.endswith(self.END_MARKER):
            print("[DEBUG] Маркер кінця не знайдено. Можливо, повідомлення неповне або пошкоджене.")

        print(f"[DEBUG] Отримане бінарне повідомлення (перші 100 біт): {binary_message[:100]}...")

        # Декодування
        try:
            text = self.msg_processor.binary_to_text(binary_message)
            print(f"[DEBUG] Витягнуте повідомлення: {text}")
            return text
        except Exception as e:
            print(f"[ERROR] Помилка декодування: {e}")
            return "Помилка декодування або повідомлення пошкоджене."

