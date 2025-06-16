# Price Analyzer

Aplikacja desktopowa do analizy wpływu cech (kolumn) na wybraną zmienną docelową w zbiorze danych (np. cena), oparta o GUI (`Tkinter`) i algorytm Random Forest.

## Funkcjonalności

- Wczytywanie pliku CSV przez okno dialogowe
- Wybór kolumny docelowej (np. cena)
- Automatyczne przetwarzanie danych (usuwanie braków, kodowanie kategorii)
- Analiza ważności cech przy użyciu Random Forest Regressor
- Wielokrotne iteracje z losowym podzbiorem cech
- Pasek postępu podczas analizy
- Wizualizacja wyników w postaci wykresu słupkowego
- Intuicyjny interfejs graficzny (`Tkinter`)

## Wymagania

- Python 3.8+
- Pakiety:
  ```bash
  pip install pandas numpy matplotlib scikit-learn
  ```

## Uruchomienie

```bash
python Price_analyzer.py
```

## Użytkowanie

1. Uruchom aplikację.
2. Kliknij "Open File" i wybierz plik CSV.
3. Wybierz kolumnę docelową z listy.
4. Kliknij "Start Analysis".
5. Po zakończeniu analizy zobaczysz wykres z wpływem poszczególnych cech.

## Autor

[GitHub](https://github.com/Dilo993)