import struct
import zlib
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from numpy.fft import fft2, fftshift

def read_chunk(file):
    """Odczytuje pojedynczy chunk z pliku PNG"""
    # Odczytanie długości i typu
    # Po odczytaniu pierwszych 8 bajtów (sygnatura PNG) odczytujemy długość chunk'a, w tym przypadku IHDR
    length_bytes = file.read(4)
    # Następnie długość chanku w hexa zmieniamy na int
    length = struct.unpack('>I', length_bytes)[0]
    
    # Odczytanie typu chunk'a (IHDR) dla pierwszego przejścia
    type_bytes = file.read(4)
    # Zamiana bajtów na string
    chunk_type = type_bytes.decode('ascii')
    
    # Odczytanie danych
    data = file.read(length)
    
    # Odczytanie CRC
    crc = file.read(4)
    
    return {
        'length': length,
        'type': chunk_type,
        'data': data,
        'crc': crc,
        'start_pos': file.tell() - length - 12
    }

def process_critical_chunks(chunks):
    """Przetwarza i wyświetla informacje z krytycznych chunków"""
    print("\n=== Informacje z krytycznych chunków ===")
    
    for chunk in chunks:
        if chunk['type'] == 'IHDR':
            print("\n[IHDR - Nagłówek obrazu]")
            # Interpretacja danych IHDR
            # Na podstawie długości pól w IHDR mamy odpowiednie ilości bajtów 4, 4, 1, 1, 1, 1, 1, a potem CRC 
            width, height, bit_depth, color_type, compression, filter_method, interlace = \
                struct.unpack('>IIBBBBB', chunk['data'])
            
            print(f"Szerokość: {width} pikseli")
            print(f"Wysokość: {height} pikseli")
            print(f"Głębia koloru: {bit_depth} bitów na kanał")
            
            color_types = {
                0: "Skala szarości",
                2: "RGB",
                3: "Paleta indeksowana",
                4: "Skala szarości z kanałem alfa",
                6: "RGB z kanałem alfa"
            }
            print(f"Typ koloru: {color_types.get(color_type, 'Nieznany')}")
            print(f"Metoda kompresji: {'DEFLATE' if compression == 0 else 'Nieznana'}")
            print(f"Metoda filtrowania: {filter_method}")
            print(f"Przeplot: {'Tak' if interlace == 1 else 'Nie'}")
        
        # PLTE będzie w zależności od typu koloru w IHDR, w przypadku 3 (paleta) będzie obecny
        elif chunk['type'] == 'PLTE':
            print("\n[PLTE - Paleta kolorów]")
            num_colors = len(chunk['data']) // 3
            print(f"Liczba kolorów w palecie: {num_colors}")
        
        elif chunk['type'] == 'IDAT':
            print("\n[IDAT - Dane obrazu]")
            print(f"Rozmiar skompresowanych danych: {chunk['length']} bajtów")
        
        elif chunk['type'] == 'IEND':
            print("\n[IEND - Koniec obrazu]")

def display_image(chunks):
    """Wyświetla obraz z uwzględnieniem filtrów i palet"""
    # Znajdź IHDR
    ihdr = next(c for c in chunks if c['type'] == 'IHDR')
    width, height, bit_depth, color_type, _, _, _ = struct.unpack('>IIBBBBB', ihdr['data'])
    
    # Zebierz dane IDAT
    idat_data = b''.join(c['data'] for c in chunks if c['type'] == 'IDAT')
    decompressed = zlib.decompress(idat_data)
    
    # Oblicz bajty na wiersz (uwzględniając filtr)
    bpp = {0:1, 
           2:3, 
           3:1, 
           4:2, 
           6:4}[color_type]
    if bit_depth == 16:
        bpp *= 2
    row_length = width * bpp
    
    # Usuń filtry i zrekonstruuj dane
    clean_data = []
    for y in range(height):
        offset = y * (row_length + 1)
        row = decompressed[offset+1 : offset+1+row_length]
        clean_data.append(row)
    clean_data = b''.join(clean_data)
    
    # Konwersja do obrazu
    try:
        mode = {0:'L', 
                2:'RGB', 
                3:'P', 
                4:'LA', 
                6:'RGBA'}[color_type]
        img = Image.frombytes(mode, (width, height), clean_data)
        
        # Dla palety (typ 3) załaduj PLTE
        if color_type == 3:
            plte = next((c for c in chunks if c['type'] == 'PLTE'), None)
            if plte:
                img.putpalette(plte['data'])
        
        img.show()
    except Exception as e:
        print(f"Błąd wyświetlania: {e}")
        

def anonymize_png(chunks, output_path):
    """Tworzy anonimizowaną wersję pliku PNG (usuwa wszystkie niekrytyczne chunki)"""
    with open(output_path, 'wb') as f:
        # Sygnatura PNG
        f.write(b'\x89PNG\r\n\x1a\n')
        
        # Zapis tylko krytycznych chunków
        for chunk in chunks:
            if chunk['type'] in ['IHDR', 'PLTE', 'IDAT', 'IEND']:
                # Długość
                f.write(struct.pack('>I', chunk['length']))
                # Typ
                f.write(chunk['type'].encode('ascii'))
                # Dane
                f.write(chunk['data'])
                # CRC
                f.write(chunk['crc'])

def compute_fft(chunks):
    ihdr = next(chunk for chunk in chunks if chunk['type'] == 'IHDR')
    width, height, bit_depth, color_type, _, _, _ = struct.unpack('>IIBBBBB', ihdr['data'])
    
    idat_data = b''.join(chunk['data'] for chunk in chunks if chunk['type'] == 'IDAT')
    try:
        decompressed_data = zlib.decompress(idat_data)
    except zlib.error as e:
        print(f"Błąd dekompresji: {e}")
        return

    # Obsługa różnych formatów
    if color_type == 0:  # Skala szarości
        try:
            img = np.frombuffer(decompressed_data, dtype=np.uint8)
            img = img.reshape((height, width + 1))[:, 1:]  # Usuń bajty filtra

            fft_img = fft2(img)
            fft_img = fftshift(fft_img)
            plt.imshow(20*np.log(np.abs(fft_img)), cmap='viridis')
            plt.title('FFT Skala szarości')
            plt.show()
            
        except ValueError as e:
            print(f"Błąd reshape dla skali szarości: {e}")
            return
            
    elif color_type == 2:  # RGB
        try:
            img = np.frombuffer(decompressed_data, dtype=np.uint8)
            img = img.reshape((height, 1 + width * 3))[:, 1:]  # Usuń bajty filtra
            img = img.reshape((height, width, 3))  # Kształt (H, W, 3)
            
            
            fft_r = fftshift(fft2(img[:,:,0]))
            fft_g = fftshift(fft2(img[:,:,1]))
            fft_b = fftshift(fft2(img[:,:,2]))
            
            # Wyświetlenie wyników
            plt.subplot(131)
            plt.imshow(20*np.log(np.abs(fft_r)), cmap='viridis')
            plt.title('Czerwony (R)')

            plt.subplot(132)
            plt.imshow(20*np.log(np.abs(fft_g)), cmap='viridis')
            plt.title('Zielony (G)')

            plt.subplot(133)
            plt.imshow(20*np.log(np.abs(fft_b)), cmap='viridis')
            plt.title('Niebieski (B)')
            plt.show()
            
            plt.show()
        except ValueError as e:
            print(f"Błąd reshape dla RGB: {e}")
            print(f"Dane: {len(img)} bajtów, Oczekiwano: ~{height}x{width}x3")
            return
    else:
        print(f"Nieobsługiwany typ koloru: {color_type}")
        return


def read_png(file_path):
    """Główna funkcja odczytująca i analizująca plik PNG"""
    with open(file_path, 'rb') as f:
        # Sprawdzenie sygnatury PNG
        signature = f.read(8)
        # Sprawdzenie czy sygnatura odpowiada plikowi PNG - 89 PNG 0D 0A 1A 0A
        if signature != b'\x89PNG\r\n\x1a\n':
            raise ValueError("To nie jest prawidłowy plik PNG")
        
        print("=== Sygnatura PNG poprawna ===")
        
        chunks = []
        while True:
            # Odczytanie kolejnego chunk'a
            chunk = read_chunk(f)
            chunks.append(chunk)
            
            # Koniec gdy napotkamy IEND
            if chunk['type'] == 'IEND':
                break
        
        # Wyświetlenie informacji o chunkach
        print("\n=== Znalezione chunki ===")
        for chunk in chunks:
            print(f"{chunk['type']} - długość: {chunk['length']} bajtów")
        
        # Przetworzenie krytycznych chunków
        process_critical_chunks(chunks)
        
        # Wyświetlenie obrazu
        display_image(chunks)
        
        # Obliczenie FFT
        compute_fft(chunks)
        
        # Anonimizacja (zapis do nowego pliku)
        anonymize_png(chunks, 'anonimized.png')
        print("\nAnonimizacja zakończona. Zapisano jako 'anonimized.png'")

# Użycie programu
if __name__ == "__main__":
    file_path = input("Podaj ścieżkę do pliku PNG: ")
    read_png(file_path)