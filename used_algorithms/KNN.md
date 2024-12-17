# K-Nearest Neighbors (KNN)

Działa na zasadzie podobieństwa między próbkami. W przypadku nowej próbki, algorytm szuka 𝑘 najbliższych sąsiadów w przestrzeni cech i przypisuje klasę na podstawie większości etykiet sąsiadów.

## Jak działa KNN?

1. Algorytm oblicza odległości między nową próbką a wszystkimi próbkami w zbiorze treningowym (np. za pomocą odległości euklidesowej).

2. Wybiera 𝑘 najbliższych sąsiadów (próbek z najmniejszą odległością).
3. Klasyfikuje nową próbkę jako tę klasę, która występuje najczęściej wśród tych sąsiadów (głosowanie większościowe).

Parametr 𝑘 (liczba sąsiadów) to kluczowy hiperparametr, który wpływa na dokładność modelu.

## Zalety i wady

Zalety:

- Prosty do zrozumienia i zaimplementowania.
- Nie wymaga "uczenia" – wystarczy zapisać dane treningowe.
- Może działać dobrze na małych zbiorach danych.

Wady:

- Wolny dla dużych zbiorów danych (dużo obliczeń w czasie predykcji).
- Wrażliwy na szum w danych (dobór 𝑘 ma kluczowe znaczenie).
- Nie działa dobrze na danych o dużej liczbie wymiarów (tzw. curse of dimensionality).