import difflib


class MessageProcessor:
    """Клас для роботи з текстовими повідомленнями - кодування, декодування"""

    @staticmethod
    def text_to_binary(text):
        """
        Перетворення тексту в бінарну послідовність

        Args:
            text (str): Текст для перетворення

        Returns:
            str: Бінарна послідовність
        """
        binary = ''.join(format(ord(i), '08b') for i in text)
        return binary

    @staticmethod
    def binary_to_text(binary):
        """
        Перетворення бінарної послідовності назад у текст

        Args:
            binary (str): Бінарна послідовність

        Returns:
            str: Відновлений текст
        """
        # Переконуємося, що довжина бінарної послідовності кратна 8
        if len(binary) % 8 != 0:
            binary = binary[:-(len(binary) % 8)]

        text = ''.join([chr(int(binary[i:i + 8], 2)) for i in range(0, len(binary), 8)])
        return text

    @staticmethod
    def calculate_ber(original_message, extracted_message):
        """
        Обчислення Bit Error Rate (BER) для порівняння оригінального та витягнутого повідомлення

        Args:
            original_message (str): Оригінальне повідомлення
            extracted_message (str): Витягнуте повідомлення

        Returns:
            float: Bit Error Rate [0.0, 1.0]
        """
        if extracted_message is None or "Не вдалося вилучити повідомлення" in extracted_message:
            return 1.0  # 100% помилок

        # Перетворюємо повідомлення в бінарний формат
        try:
            msg_processor = MessageProcessor()
            original_binary = msg_processor.text_to_binary(original_message)

            # Вирівнюємо довжини рядків для порівняння
            max_len = min(len(original_binary), len(extracted_message) * 8)
            original_binary = original_binary[:max_len]

            # Якщо extracted_message вже бінарний
            if all(c in '01' for c in extracted_message):
                extracted_binary = extracted_message[:max_len]
            else:
                extracted_binary = msg_processor.text_to_binary(extracted_message)[:max_len]

            # Підрахунок кількості помилкових бітів
            errors = sum(1 for a, b in zip(original_binary, extracted_binary) if a != b)

            return errors / len(original_binary)
        except:
            # Якщо виникла помилка при перетворенні, використовуємо інший підхід через відстань Левенштейна
            sequence_matcher = difflib.SequenceMatcher(None, original_message, extracted_message)
            similarity = sequence_matcher.ratio()
            return 1.0 - similarity
