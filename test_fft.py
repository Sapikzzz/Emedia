import numpy as np

def test_fft_correctness():
    """Proponowany sposób testowania poprawności transformacji Fouriera."""
    print("\n=== Testowanie Transformacji Fouriera ===")

    # Test 1: Jednolity obraz (oczekiwane widmo: pojedynczy punkt)
    print("Test 1: Obraz jednolity...")
    uniform_img = np.full((64, 64), 128, dtype=np.uint8) # Szary obraz
    fft_uniform = np.fft.fft2(uniform_img)
    fft_uniform_shifted = np.fft.fftshift(fft_uniform)
    magnitude_uniform = 20 * np.log(np.abs(fft_uniform_shifted) + 1e-9) # Dodano małą wartość, aby uniknąć log(0)

    # Sprawdzenie, czy dominująca jest składowa stała (DC component)
    max_val_idx = np.unravel_index(np.argmax(magnitude_uniform), magnitude_uniform.shape)
    center_idx = (magnitude_uniform.shape[0]//2, magnitude_uniform.shape[1]//2)
    if max_val_idx == center_idx:
        print("  PASS: Dominująca składowa stała w centrum widma.")
    else:
        print(f"  FAIL: Dominująca składowa stała poza centrum ({max_val_idx} zamiast {center_idx}).")

    # Można wizualizować, jeśli chcemy potwierdzić:
    # plt.imshow(magnitude_uniform, cmap='viridis'); plt.title('FFT - Obraz Jednolity'); plt.show()

    # Test 2: Odwrotna transformacja (porównanie z oryginałem)
    print("Test 2: Odwrotna transformacja Fouriera...")
    # Użyjemy prostego obrazu testowego
    test_image_data = np.array([
        [0, 10, 20],
        [30, 40, 50],
        [60, 70, 80]
    ], dtype=np.uint8)

    # Oblicz FFT
    fft_result = np.fft.fft2(test_image_data)
    # Oblicz IFFT
    ifft_result = np.fft.ifft2(fft_result)

    # Wynik IFFT powinien być bliski oryginalnemu obrazowi (z uwagi na błędy numeryczne, porównujemy z tolerancją)
    if np.allclose(test_image_data, np.abs(ifft_result)):
        print("  PASS: Obraz odtworzony z IFFT jest zgodny z oryginałem.")
    else:
        print("  FAIL: Obraz odtworzony z IFFT różni się od oryginału.")
        # print("Oryginał:\n", test_image_data)
        # print("Odtworzony (abs):\n", np.abs(ifft_result))

    # Test 3: Symetryczność widma dla danych rzeczywistych
    print("Test 3: Symetryczność widma...")
    # Wykorzystajmy magnitude_uniform z Testu 1
    # Sprawdź symetrię względem środka
    if magnitude_uniform.shape[0] % 2 != 0 or magnitude_uniform.shape[1] % 2 != 0:
        print("  UWAGA: Obraz ma nieparzyste wymiary, symetria może być trudniejsza do bezpośredniego sprawdzenia.")
    
    # Przykładowe sprawdzenie symetrii dla parzystych wymiarów
    # Porównaj lewą stronę z odwróconą prawą itp.
    # To jest uproszczone; pełne sprawdzenie symetrii jest bardziej złożone
    # Np. abs(F(u,v)) == abs(F(-u,-v))
    # Dla np.fft.fftshift(fft_img), środek jest w (N/2, M/2)
    rows, cols = magnitude_uniform.shape
    if np.allclose(magnitude_uniform[rows//2:, cols//2:], np.flip(magnitude_uniform[:rows//2, :cols//2], axis=(0,1))):
        print("  PASS: Widmo wykazuje symetrię (sprawdzone fragmenty).")
    else:
        print("  FAIL: Widmo nie wykazuje symetrii.")