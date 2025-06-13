# png_handler.py

import struct
import zlib

def _read_chunk(file):
    """Odczytuje pojedynczy chunk z pliku PNG (funkcja pomocnicza)."""
    try:
        length_bytes = file.read(4)
        if not length_bytes:
            return None
        length = struct.unpack('>I', length_bytes)[0]
        
        type_bytes = file.read(4)
        chunk_type = type_bytes.decode('ascii')
        
        data = file.read(length)
        crc = file.read(4)
        
        return {'length': length, 'type': chunk_type, 'data': data, 'crc': crc}
    except (struct.error, IndexError):
        return None

def read_png_file(file_path):
    """Odczytuje plik PNG, sprawdza sygnaturę i zwraca listę chunków."""
    with open(file_path, 'rb') as f:
        if f.read(8) != b'\x89PNG\r\n\x1a\n':
            raise ValueError("To nie jest prawidłowy plik PNG")
        
        print("=== Sygnatura PNG poprawna ===")
        chunks = []
        while True:
            chunk = _read_chunk(f)
            if chunk is None or chunk['type'] == 'IEND':
                if chunk:
                    chunks.append(chunk)
                break
            chunks.append(chunk)
    return chunks

def print_critical_chunks_info(chunks):
    """Przetwarza i wyświetla informacje z krytycznych chunków."""
    print("\n=== Informacje z krytycznych chunków ===")
    for chunk in chunks:
        if chunk['type'] == 'IHDR':
            # ... (cała logika z oryginalnej funkcji process_critical_chunks dla IHDR) ...
            width, height, bit_depth, color_type, compression, filter_method, interlace = \
                struct.unpack('>IIBBBBB', chunk['data'])
            # ... itd.
            print("\n[IHDR - Nagłówek obrazu]")
            print(f"Szerokość: {width} pikseli")
            # ... reszta printów
            
        elif chunk['type'] == 'PLTE':
            print("\n[PLTE - Paleta kolorów]")
            # ... reszta logiki
            
        elif chunk['type'] == 'IDAT':
            print(f"\n[IDAT] - Rozmiar skompresowanych danych: {chunk['length']} bajtów")
            
        elif chunk['type'] == 'IEND':
            print("\n[IEND - Koniec obrazu]")


def anonymize_png(chunks, output_path):
    """Tworzy anonimizowaną wersję pliku PNG (usuwa wszystkie niekrytyczne chunki)."""
    with open(output_path, 'wb') as f:
        f.write(b'\x89PNG\r\n\x1a\n')
        for chunk in chunks:
            if chunk['type'] in ['IHDR', 'PLTE', 'IDAT', 'IEND']:
                f.write(struct.pack('>I', chunk['length']))
                f.write(chunk['type'].encode('ascii'))
                f.write(chunk['data'])
                f.write(chunk['crc'])
    print(f"\nAnonimizacja zakończona. Zapisano jako '{output_path}'")