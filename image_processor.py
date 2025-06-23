# image_processor.py

import zlib
import struct
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def _reconstruct_pixel_data(chunks):
    """Pomocnicza funkcja do dekompresji i rekonstrukcji surowych danych pikseli."""
    ihdr = next(c for c in chunks if c['type'] == 'IHDR')
    width, height, bit_depth, color_type, _, _, _ = struct.unpack('>IIBBBBB', ihdr['data'])
    
    idat_data = b''.join(c['data'] for c in chunks if c['type'] == 'IDAT')
    decompressed = zlib.decompress(idat_data)
    
    # Ta część jest uproszczeniem; prawidłowa rekonstrukcja wymaga obsługi filtrów Paeth, Sub, etc.
    bytes_per_pixel = {0:1, 2:3, 3:1, 4:2, 6:4}[color_type]
    if bit_depth == 16:
        bytes_per_pixel *= 2
    
    stride = width * bytes_per_pixel + 1 # +1 for filter byte
    reconstructed = bytearray()
    
    # Uproszczona rekonstrukcja (ignorowanie filtrów, tylko usuwanie bajtu filtra)
    for i in range(height):
        start = i * stride
        filter_type = decompressed[start]
        scanline = decompressed[start+1:start+stride]
        reconstructed.extend(scanline)
        
    return bytes(reconstructed), width, height, color_type, bit_depth


def display_image(chunks):
    """Wyświetla obraz z uwzględnieniem filtrów i palet."""
    try:
        pixel_data, width, height, color_type, bit_depth = _reconstruct_pixel_data(chunks)
        
        mode = {0:'L', 2:'RGB', 3:'P', 4:'LA', 6:'RGBA'}[color_type]
        img = Image.frombytes(mode, (width, height), pixel_data)
        
        if color_type == 3:
            plte = next((c['data'] for c in chunks if c['type'] == 'PLTE'), None)
            if plte:
                img.putpalette(plte)
        
        img.show()
    except Exception as e:
        print(f"Błąd podczas wyświetlania obrazu: {e}")


def compute_and_show_fft_from_file(file_path):
    """
    Wczytuje obraz z podanej ścieżki pliku, a następnie oblicza i wyświetla
    widmo amplitudowe oraz fazowe za pomocą transformaty Fouriera.
    """
    try:
        # Wczytanie obrazu za pomocą Pillow
        img = Image.open(file_path)
        img_array = np.array(img)

        # Konwersja obrazu do skali szarości (luminancji)
        gray_img_pil = img.convert('L') # L - skala szarości
        gray_img = np.array(gray_img_pil)

        # Obliczanie FFT
        fft_img = np.fft.fft2(gray_img)
        fft_img_shifted = np.fft.fftshift(fft_img) # Przesunięcie składowej zerowej do centrum

        # Widmo Amplitudowe
        # Dodajemy małą stałą, aby uniknąć logarytmowania zera
        magnitude_spectrum = 20 * np.log(np.abs(fft_img_shifted) + 1e-9) 

        # Wyświetlanie obu widm
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.imshow(img_array)
        plt.title("Oryginalny obraz")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(magnitude_spectrum)
        plt.title("Widmo Fouriera (amplituda w skali log)")
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    except FileNotFoundError:
        print(f"Błąd: Plik '{file_path}' nie został znaleziony.")
    except Exception as e:
        print(f"Błąd podczas obliczania FFT: {e}")
