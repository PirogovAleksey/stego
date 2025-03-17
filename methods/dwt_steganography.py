import numpy as np
import pywt


class DWTSteganographyFloat32:
    """Клас для стеганографії з використанням DWT з підтримкою float32"""

    # Константи класу
    END_MARKER = '1111111111111110'  # Маркер кінця повідомлення

    def __init__(self, audio_reader, msg_processor):
        self.audio_reader = audio_reader
        self.msg_processor = msg_processor
        self.wavelet = 'db1'  # Тип вейвлета
        self.level = 5  # Рівень декомпозиції
        self.subband_idx = 2  # Індекс піддіапазону (2 відповідає cD2)
        self.step = 100  # Крок між коефіцієнтами
        self.bit1_frac = 0.75  # Дробова частина для біту 1
        self.bit0_frac = 0.25  # Дробова частина для біту 0

    def embed(self, audio_path, message, output_path):
        """
        Вбудовування даних в аудіофайл за допомогою DWT з float32

        Args:
            audio_path (str): Шлях до оригінального аудіофайлу
            message (str): Повідомлення для вбудовування
            output_path (str): Шлях для збереження результату

        Returns:
            tuple: (оригінальні дані, модифіковані дані)
        """
        # Зчитування аудіофайлу у float32
        sample_rate, audio_data = self.audio_reader.read_audio(audio_path)

        # Вибір каналу для стерео файлів
        if audio_data.ndim == 2:
            print("[DEBUG] Стерео виявлено, працюємо з першим каналом")
            channel_1 = audio_data[:, 0].copy()
        else:
            channel_1 = audio_data.copy()

        # Підготовка повідомлення
        binary_message = self.msg_processor.text_to_binary(message) + self.END_MARKER
        print(f"[DEBUG] Бінарне повідомлення для вбудовування ({len(binary_message)} біт): {binary_message[:50]}...")

        # Застосування DWT
        coeffs = pywt.wavedec(channel_1, self.wavelet, level=self.level)
        target_subband = coeffs[self.subband_idx]

        print(f"[DEBUG] Довжина піддіапазону: {len(target_subband)}")
        print(f"[DEBUG] Доступна кількість місць: {len(target_subband) // self.step}")

        if len(binary_message) > len(target_subband) // self.step:
            raise ValueError(f"Повідомлення занадто велике. Максимум {len(target_subband) // self.step} біт.")

        # Вбудовування бітів повідомлення
        for i, bit in enumerate(binary_message):
            if i * self.step >= len(target_subband):
                break

            idx = i * self.step

            # Встановлення дробової частини коефіцієнта
            base = np.floor(target_subband[idx])
            if bit == '1':
                target_subband[idx] = base + self.bit1_frac
            else:
                target_subband[idx] = base + self.bit0_frac

        print(f"[DEBUG] Вбудовано {len(binary_message)} бітів")

        # Зворотне DWT
        modified_channel = pywt.waverec(coeffs, self.wavelet)

        # Обрізаємо до оригінальної довжини
        modified_channel = modified_channel[:len(channel_1)]

        # Створення модифікованого аудіо
        if audio_data.ndim == 2:
            modified_audio = audio_data.copy()
            modified_audio[:, 0] = modified_channel
        else:
            modified_audio = modified_channel

        # Збереження результату у float32
        self.audio_reader.save_audio(output_path, sample_rate, modified_audio)
        print(f"[DEBUG] Повідомлення вбудовано з використанням DWT, піддіапазон: {self.subband_idx}")

        return audio_data, modified_audio

    def extract(self, audio_path):
        """
        Вилучення даних з аудіофайлу за допомогою DWT

        Args:
            audio_path (str): Шлях до аудіофайлу зі стеганограмою

        Returns:
            str: Вилучене повідомлення
        """
        # Зчитування аудіофайлу
        sample_rate, audio_data = self.audio_reader.read_audio(audio_path)

        # Вибір каналу для стерео файлів
        if audio_data.ndim == 2:
            print("[DEBUG] Стерео виявлено, працюємо з першим каналом")
            channel_1 = audio_data[:, 0]
        else:
            channel_1 = audio_data

        # Застосування DWT
        coeffs = pywt.wavedec(channel_1, self.wavelet, level=self.level)
        target_subband = coeffs[self.subband_idx]

        binary_message = ""

        # Виводимо діагностику перших коефіцієнтів
        print("[DEBUG] Аналіз перших 5 коефіцієнтів:")
        for i in range(5):
            idx = i * self.step
            if idx < len(target_subband):
                frac_part = target_subband[idx] - np.floor(target_subband[idx])
                print(f"Індекс {idx}, Дробова частина: {frac_part:.6f} -> {'1' if frac_part >= 0.5 else '0'}")

        # Вилучення бітів повідомлення
        for i in range(len(target_subband) // self.step):
            idx = i * self.step

            # Вилучення біту на основі дробової частини
            frac_part = target_subband[idx] - np.floor(target_subband[idx])

            if frac_part >= 0.5:
                binary_message += '1'
            else:
                binary_message += '0'

            # Перевірка на маркер кінця повідомлення
            if binary_message.endswith(self.END_MARKER):
                binary_message = binary_message[:-len(self.END_MARKER)]
                print(f"[DEBUG] Маркер кінця знайдено на позиції {i}")
                break
        else:
            print("[DEBUG] Маркер кінця не знайдено!")

        # Виведення діагностики
        print(f"[DEBUG] Довжина вилученого бінарного повідомлення: {len(binary_message)} біт")
        if len(binary_message) > 0:
            print(f"[DEBUG] Початок бінарного повідомлення: {binary_message[:50]}...")
        else:
            return "Не вдалося вилучити повідомлення"

        # Перетворення бінарного повідомлення назад у текст
        try:
            text = self.msg_processor.binary_to_text(binary_message)
            return text
        except Exception as e:
            print(f"[DEBUG] Помилка при декодуванні: {e}")
            return f"Помилка декодування: {e}"
