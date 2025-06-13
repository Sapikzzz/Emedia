# main.py

import sys
import png_handler
import image_processor

def main():
    """Główna funkcja programu."""
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = input("Podaj ścieżkę do pliku PNG: ")
        
    try:
        # 1. Wczytaj i przeanalizuj plik PNG
        chunks = png_handler.read_png_file(file_path)
        
        # 2. Wyświetl informacje o chunkach
        print("\n=== Znalezione chunki ===")
        print(", ".join([chunk['type'] for chunk in chunks]))
        png_handler.print_critical_chunks_info(chunks)
        
        # 3. Wyświetl obraz
        print("\nPróba wyświetlenia obrazu...")
        image_processor.display_image(chunks)
        
        # 4. Oblicz i wyświetl FFT
        print("Obliczanie transformaty Fouriera...")
        image_processor.compute_and_show_fft(chunks)
        
        # 5. Dokonaj anonimizacji
        output_path = 'anonymized.png'
        png_handler.anonymize_png(chunks, output_path)

    except FileNotFoundError:
        print(f"Błąd: Plik '{file_path}' nie został znaleziony.")
    except ValueError as e:
        print(f"Błąd przetwarzania pliku PNG: {e}")
    except Exception as e:
        print(f"Wystąpił nieoczekiwany błąd: {e}")

if __name__ == "__main__":
    main()