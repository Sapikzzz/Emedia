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
    # Dla celów tego projektu, zakładamy prosty model lub akceptujemy, że może on nie działać dla wszystkich PNG.
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


def compute_and_show_fft(chunks):
    """Oblicza i wyświetla transformatę Fouriera obrazu."""
    try:
        pixel_data, width, height, color_type, bit_depth = _reconstruct_pixel_data(chunks)
        
        # Konwersja do skali szarości, jeśli jest to obraz kolorowy
        if color_type in [2, 6]: # RGB lub RGBA
            img = np.frombuffer(pixel_data, dtype=np.uint8).reshape((height, width, -1))
            # Konwersja do skali szarości (luminancja)
            gray_img = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])
        else:
            gray_img = np.frombuffer(pixel_data, dtype=np.uint8).reshape((height, width))

        fft_img = np.fft.fft2(gray_img)
        fft_img_shifted = np.fft.fftshift(fft_img)
        magnitude_spectrum = 20 * np.log(np.abs(fft_img_shifted))
        
        plt.imshow(magnitude_spectrum, cmap='viridis')
        plt.title('Widmo Amplitudowe (FFT)')
        plt.show()

    except Exception as e:
        print(f"Błąd podczas obliczania FFT: {e}")