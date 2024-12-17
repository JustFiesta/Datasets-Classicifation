# Naive Bayes

Prosty i skuteczny algorytm klasyfikacji oparty na twierdzeniu Bayesa, który zakłada niezależność cech (tzw. naive assumption). Mimo tego założenia, działa bardzo dobrze w praktyce, szczególnie w zadaniach klasyfikacji tekstu, takich jak filtrowanie spamu.

## Jak działa Naive Bayes?

Naive Bayes oblicza prawdopodobieństwo przynależności próbki do określonej klasy na podstawie cech. W przypadku klasyfikacji tekstu, algorytm opiera się na częstości występowania słów.

Formuła:

`P(A∣B)= P(B) / P(B∣A)⋅P(A)`

- 𝑃(𝑦∣𝑋): Prawdopodobieństwo, że próbka należy do klasy 𝑦,
biorąc pod uwagę cechy 𝑋.
- 𝑃(𝑋∣𝑦): Prawdopodobieństwo zaobserwowania cech 𝑋 w klasie 𝑦.
- 𝑃(𝑦): Prawdopodobieństwo wystąpienia klasy 𝑦.
- 𝑃(𝑋): Prawdopodobieństwo wystąpienia cech 𝑋 (pomijamy, ponieważ jest stałe dla wszystkich klas).

Najczęściej używanym wariantem jest Multinomial Naive Bayes, który zakłada, że cechy to liczniki (np. liczba wystąpień słów w dokumencie).

## Zalety i wady

Zalety:

- Bardzo szybki w trenowaniu i przewidywaniu.
- Dobrze działa na dużych zbiorach danych.
- Szczególnie skuteczny w klasyfikacji tekstu.

Wady:

- Założenie niezależności cech rzadko jest spełnione.
- Może działać gorzej na danych liczbowych (ciągłych).
