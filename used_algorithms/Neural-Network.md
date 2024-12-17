# Sieci neuronowe

Algorytm inspirowany działaniem ludzkiego mózgu. Jest niezwykle potężny w klasyfikacji i regresji, szczególnie dla złożonych danych. Działa poprzez łączenie warstw neuronów, które przekształcają dane wejściowe, by nauczyć się wzorców i podejmować decyzje.

W klasyfikacji problemu spam/ham możemy użyć Multi-Layer Perceptron (MLP) – jednego z podstawowych modeli sieci neuronowych. W Scikit-learn model ten jest zaimplementowany w klasie MLPClassifier.

## Kluczowe koncepcje

1. Warstwy ukryte – przekształcają dane wejściowe.
2. Funkcje aktywacji – dodają nieliniowość do modelu (np. ReLU, sigmoid).
3. Epoki i iteracje – sieć uczy się, aktualizując wagi wielokrotnie na podstawie danych.
4. Optymalizacja – metoda dostrajania wag modelu (np. algorytm Adam).

## Zalety i wady

Zalety:

- Potężny model dla złożonych, nieliniowych danych.
- Możliwość dostosowania struktury (liczba warstw i neuronów).
- Dobrze działa na dużych zbiorach danych.

Wady:

- Wymaga dużej liczby danych do skutecznego trenowania.
- Wymaga odpowiedniego dostrajania parametrów (np. liczby warstw, neuronów).
- Może być wolny w przypadku bardzo złożonych konfiguracji.
