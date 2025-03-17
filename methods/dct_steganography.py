import numpy as np
from scipy.fftpack import dct, idct


class DCTSteganographyFloat32:
    """Клас для стеганографії з використанням DCT з підтримкою float32"""

    # Константи класу
    END_MARKER = '1111111111111110'  # Маркер кінця повідомлення

    def __init__(self, audio_reader, msg_processor):
        self.audio_reader = audio_reader
        self.msg_processor = msg_processor
        self.block_size = 1024  # Розмір блоку для DCT
        self.target_coef = 5  # Коефіцієнт для вбудовування
        self.bit1_value = 0.1  # Значення для біту 1
        self.bit0_value = -0.1  # Значення для біту 0
        self.threshold = 0.0  # Поріг для розпізнавання бітів

    def embed(self, audio_path, message, output_path):
        """
        Вбудовування даних в аудіофайл за допомогою DCT з float32

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

        # Розбиття аудіо на блоки
        num_blocks = len(channel_1) // self.block_size
        print(f"[DEBUG] Кількість блоків: {num_blocks}, Розмір блоку: {self.block_size}")

        if len(binary_message) > num_blocks:
            raise ValueError(f"Повідомлення занадто велике. Максимум {num_blocks} біт.")

        modified_channel = channel_1.copy()

        # Вбудовування бітів повідомлення
        for i in range(len(binary_message)):
            if i >= num_blocks:
                break

            # Виділення блоку
            start_idx = i * self.block_size
            end_idx = start_idx + self.block_size
            block = channel_1[start_idx:end_idx]

            # Виконання DCT
            block_dct = dct(block, type=2, norm='ortho')

            # Вбудовування біту повідомлення з використанням фіксованих значень
            if binary_message[i] == '1':
                block_dct[self.target_coef] = self.bit1_value
            else:
                block_dct[self.target_coef] = self.bit0_value

            # Інверсне DCT
            modified_block = idct(block_dct, type=2, norm='ortho')

            # Заміна блоку в аудіо
            modified_channel[start_idx:end_idx] = modified_block

        # Створення модифікованого аудіо
        if audio_data.ndim == 2:
            modified_audio = audio_data.copy()
            modified_audio[:, 0] = modified_channel
        else:
            modified_audio = modified_channel

        # Збереження результату у float32
        self.audio_reader.save_audio(output_path, sample_rate, modified_audio)
        print(f"[DEBUG] Повідомлення вбудовано з використанням DCT, цільовий коефіцієнт: {self.target_coef}")

        return audio_data, modified_audio

    def extract(self, audio_path):
        """
        Вилучення даних з аудіофайлу із використанням DCT

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

        # Розбиття аудіо на блоки
        block_size = self.block_size
        num_blocks = len(channel_1) // block_size

        binary_message = ""

        # Виводимо діагностику перших блоків
        print("[DEBUG] Аналіз перших 5 блоків:")
        for i in range(min(5, num_blocks)):
            start_idx = i * block_size
            end_idx = start_idx + block_size
            block = channel_1[start_idx:end_idx]

            block_dct = dct(block, type=2, norm='ortho')
            coef_value = block_dct[self.target_coef]

            print(
                f"Блок {i}, Коефіцієнт {self.target_coef}: {coef_value:.6f} -> {'1' if coef_value > self.threshold else '0'}")

        # Вилучення бітів повідомлення
        for i in range(num_blocks):
            # Виділення блоку
            start_idx = i * block_size
            end_idx = start_idx + block_size
            block = channel_1[start_idx:end_idx]

            # Виконання DCT
            block_dct = dct(block, type=2, norm='ortho')

            # Вилучення біту
            coef_value = block_dct[self.target_coef]
            if coef_value > self.threshold:
                binary_message += '1'
            else:
                binary_message += '0'

            # Перевірка на маркер кінця повідомлення
            if binary_message.endswith(self.END_MARKER):
                binary_message = binary_message[:-len(self.END_MARKER)]
                print(f"[DEBUG] Маркер кінця знайдено на позиції {i}")
                break

        # Виведення діагностики
        print(f"[DEBUG] Довжина бінарного повідомлення: {len(binary_message)} біт")
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
            return f"Помилка декодування: {e}. Бінарний рядок: {binary_message[:50]}..."
