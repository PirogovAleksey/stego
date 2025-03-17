from abc import ABC, abstractmethod


class SteganographyBase(ABC):
    """Базовий абстрактний клас для всіх методів стеганографії"""

    def __init__(self, audio_reader, msg_processor):
        """
        Ініціалізація базового класу

        Args:
            audio_reader: Об'єкт для роботи з аудіо
            msg_processor: Об'єкт для роботи з повідомленнями
        """
        self.audio_reader = audio_reader
        self.msg_processor = msg_processor

    @abstractmethod
    def embed(self, audio_path, message, output_path):
        """
        Вбудовування даних в аудіофайл

        Args:
            audio_path (str): Шлях до оригінального аудіофайлу
            message (str): Повідомлення для вбудовування
            output_path (str): Шлях для збереження результату

        Returns:
            tuple: (оригінальні дані, модифіковані дані)
        """
        pass

    @abstractmethod
    def extract(self, audio_path):
        """
        Вилучення даних з аудіофайлу

        Args:
            audio_path (str): Шлях до аудіофайлу зі стеганограмою

        Returns:
            str: Вилучене повідомлення
        """
        pass

    def prepare_message(self, message, with_marker=True):
        """
        Підготовка повідомлення для вбудовування

        Args:
            message (str): Оригінальне повідомлення
            with_marker (bool): Чи додавати маркер кінця

        Returns:
            str: Бінарне повідомлення готове для вбудовування
        """
        binary_message = self.msg_processor.text_to_binary(message)
        if with_marker:
            binary_message += '1111111111111110'  # Маркер кінця повідомлення
        return binary_message

    def process_extracted_binary(self, binary_message):
        """
        Обробка вилученого бінарного повідомлення

        Args:
            binary_message (str): Вилучене бінарне повідомлення

        Returns:
            str: Текстове повідомлення
        """
        # Перевірка на маркер кінця та видалення його
        marker_pos = binary_message.find('1111111111111110')
        if marker_pos != -1:
            binary_message = binary_message[:marker_pos]

        # Переконуємося, що довжина кратна 8
        remainder = len(binary_message) % 8
        if remainder != 0:
            binary_message = binary_message[:-remainder]

        try:
            return self.msg_processor.binary_to_text(binary_message)
        except Exception as e:
            return f"Помилка декодування: {str(e)}"
