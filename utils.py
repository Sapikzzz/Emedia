import zlib
import numpy as np
import struct

def parse_itxt_chunk_data(chunk_data):
    """
    Parsuje dane chunku iTXt zgodnie ze specyfikacją PNG.
    Zwraca słownik z sparsowanymi danymi.
    Podnosi ValueError, jeśli dane są nieprawidłowe.
    """
    offset = 0

    # 1. Keyword (Latin-1, zakończony null)
    try:
        keyword_end = chunk_data.find(b'\x00', offset)
        if keyword_end == -1:
            raise ValueError("Brak zakończenia Keyword")
        # Keyword jest kodowany w Latin-1 (ISO/IEC 8859-1)
        keyword = chunk_data[offset:keyword_end].decode('latin-1')
        offset = keyword_end + 1
    except UnicodeDecodeError:
        keyword = "[BŁĄD DEKODOWANIA KEYWORD - nieprawidłowe Latin-1]"

    # 2. Compression Flag (1 bajt)
    if offset >= len(chunk_data):
        raise ValueError("Brak Compression Flag w danych iTXt")
    compression_flag = chunk_data[offset]
    offset += 1

    # 3. Compression Method (1 bajt)
    if offset >= len(chunk_data):
        raise ValueError("Brak Compression Method w danych iTXt")
    compression_method = chunk_data[offset]
    offset += 1

    # 4. Language Tag (UTF-8, zakończony null)
    try:
        lang_tag_end = chunk_data.find(b'\x00', offset)
        if lang_tag_end == -1:
            raise ValueError("Brak zakończenia Language Tag w danych iTXt")
        lang_tag = chunk_data[offset:lang_tag_end].decode('utf-8')
        offset = lang_tag_end + 1
    except UnicodeDecodeError:
        lang_tag = "[BŁĄD DEKODOWANIA LANGUAGE TAG - nieprawidłowe UTF-8]"

    # 5. Translated Keyword (UTF-8, zakończony null)
    try:
        translated_keyword_end = chunk_data.find(b'\x00', offset)
        if translated_keyword_end == -1:
            raise ValueError("Brak zakończenia Translated Keyword w danych iTXt")
        translated_keyword = chunk_data[offset:translated_keyword_end].decode('utf-8')
        offset = translated_keyword_end + 1
    except UnicodeDecodeError:
        translated_keyword = "[BŁĄD DEKODOWANIA TRANSLATED KEYWORD - nieprawidłowe UTF-8]"

    # 6. Text (UTF-8, opcjonalnie skompresowany)
    text_data = chunk_data[offset:]

    decoded_text = ""
    if compression_flag == 1:
        if compression_method == 0:  # Zlib (deflate)
            try:
                decoded_text = zlib.decompress(text_data).decode('utf-8')
            except zlib.error as e:
                decoded_text = f"[Błąd dekompresji tekstu: {e}]"
            except UnicodeDecodeError:
                decoded_text = "[Błąd dekodowania zdekompresowanego tekstu - nieprawidłowe UTF-8]"
        else:
            decoded_text = f"[Skompresowany tekst - nieznana metoda kompresji: {compression_method}]"
    else:
        try:
            decoded_text = text_data.decode('utf-8')
        except UnicodeDecodeError:
            decoded_text = "[Błąd dekodowania nieskompresowanego tekstu - nieprawidłowe UTF-8]"

    return {
        'keyword': keyword,
        'compression_flag': compression_flag,
        'compression_method': compression_method,
        'language_tag': lang_tag,
        'translated_keyword': translated_keyword,
        'text': decoded_text
    }
    
def generate_palette_image_numpy(palette_data, width=32):
    """Generuje tablicę NumPy reprezentującą obraz z danych palety."""
    num_entries = len(palette_data) // 3
    height = (num_entries + width - 1) // width  # Oblicz wysokość obrazu palety

    # Tworzymy pustą tablicę NumPy o wymiarach (wysokość, szerokość, 3) dla RGB
    # Typ danych to uint8 (nieujemne liczby całkowite od 0 do 255)
    img_array = np.zeros((height, width, 3), dtype=np.uint8)

    for i in range(num_entries):
        x = i % width
        y = i // width
        r = palette_data[i * 3]
        g = palette_data[i * 3 + 1]
        b = palette_data[i * 3 + 2]
        img_array[y, x] = [r, g, b] # Ustawiamy piksel w tablicy NumPy
    return img_array

def parse_ihdr_chunk(ihdr_data):
    """
    Parsuje dane chunku IHDR.
    Zwraca słownik z kluczowymi informacjami: width, height, bit_depth, color_type.
    """
    if len(ihdr_data) != 13:
        raise ValueError("IHDR chunk data has incorrect length (expected 13 bytes)")

    width, height, bit_depth, color_type, compression_method, filter_method, interlace_method = struct.unpack('>IIBBBBB', ihdr_data)
    
    return {
        'width': width,
        'height': height,
        'bit_depth': bit_depth,
        'color_type': color_type,
        'compression_method': compression_method,
        'filter_method': filter_method,
        'interlace_method': interlace_method
    }