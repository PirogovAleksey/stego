import os
from utils.audio_reader import AudioReader
from utils.message_processor import MessageProcessor
from methods.dft_steganography import DFTSteganographyFloat32
from methods.dct_steganography import DCTSteganographyFloat32
from methods.dwt_steganography import DWTSteganographyFloat32


def main():
    """
    Головна функція для демонстрації роботи системи аудіо-стеганографії з float32
    """
    # Створення каталогів для результатів
    results_dir = 'results'
    steganography_dir = os.path.join(results_dir, 'steganography')

    for directory in [results_dir, steganography_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # Шлях до вхідного аудіофайлу
    input_audio = "data/file_example_MP3_5MG.mp3"

    # Повідомлення для приховування
    secret_message = "New Message Test Test"

    # Створення основних об'єктів
    audio_reader = AudioReader()
    msg_processor = MessageProcessor()

    # Створення стеганографічних методів з float32
    dft_stego = DFTSteganographyFloat32(audio_reader, msg_processor)
    dct_stego = DCTSteganographyFloat32(audio_reader, msg_processor)
    dwt_stego = DWTSteganographyFloat32(audio_reader, msg_processor)

    try:
        # Тестування DFT методу
        print("\n===== Демонстрація методу DFT з float32 =====")
        dft_output = os.path.join(steganography_dir, "output_dft_float32.wav")
        dft_stego.embed(input_audio, secret_message, dft_output)

        # Вилучення повідомлення
        extracted_dft = dft_stego.extract(dft_output)
        print(f"DFT float32: Вилучене повідомлення: {extracted_dft}")
        print(f"DFT float32: Успіх: {extracted_dft == secret_message}")

        # Тестування DCT методу
        print("\n===== Демонстрація методу DCT з float32 =====")
        dct_output = os.path.join(steganography_dir, "output_dct_float32.wav")
        dct_stego.embed(input_audio, secret_message, dct_output)

        # Вилучення повідомлення
        extracted_dct = dct_stego.extract(dct_output)
        print(f"DCT float32: Вилучене повідомлення: {extracted_dct}")
        print(f"DCT float32: Успіх: {extracted_dct == secret_message}")

        # Тестування DWT методу
        print("\n===== Демонстрація методу DWT з float32 =====")
        dwt_output = os.path.join(steganography_dir, "output_dwt_float32.wav")
        dwt_stego.embed(input_audio, secret_message, dwt_output)

        # Вилучення повідомлення
        extracted_dwt = dwt_stego.extract(dwt_output)
        print(f"DWT float32: Вилучене повідомлення: {extracted_dwt}")
        print(f"DWT float32: Успіх: {extracted_dwt == secret_message}")

        # Порівняння методів
        print("\n===== Порівняння методів float32 =====")
        methods = ["DFT", "DCT", "DWT"]
        extracts = [extracted_dft, extracted_dct, extracted_dwt]
        success = [m == secret_message for m in extracts]

        print(f"{'Метод':<10} | {'Успіх':<10} | {'Вилучене повідомлення'}")
        print("-" * 60)
        for i, method in enumerate(methods):
            print(f"{method:<10} | {str(success[i]):<10} | {extracts[i][:30]}...")

    except Exception as e:
        print(f"Помилка під час виконання: {str(e)}")


if __name__ == "__main__":
    main()
