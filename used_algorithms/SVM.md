# Support Vector Machines (SVM) 

Wszechstronny algorytm klasyfikacji. Działa poprzez znalezienie optymalnej "hiperpłaszczyzny" w przestrzeni cech, która maksymalnie oddziela klasy. Jest szczególnie skuteczny w przypadku danych o wyraźnej separowalności oraz w zadaniach o wysokiej liczbie wymiarów.

## Jak działa SVM?

1. Hiperpłaszczyzna:
    Algorytm szuka granicy decyzyjnej, która najlepiej rozdziela klasy, maksymalizując margines – odległość między granicą a najbliższymi punktami z każdej klasy (tzw. wektorami nośnymi, ang. support vectors).
2. Jądra (kernels):
    Jeśli dane nie są liniowo separowalne, SVM może użyć funkcji jądra (np. RBF, wielomianowego), aby przekształcić dane w wyższy wymiar, gdzie separacja jest możliwa.
3. Regularizacja:
    SVM wykorzystuje parametr 𝐶 do równoważenia precyzji granicy decyzyjnej i tolerancji błędów klasyfikacji.

## Zalety i wady

Zalety:

- Skuteczny w wysokowymiarowych danych.
- Stabilny w przypadku szumu w danych.
- Obsługuje nieliniowe separacje dzięki funkcjom jądra.

Wady:

- Wymaga skalowania danych (np. używając StandardScaler).
- Wolny dla bardzo dużych zbiorów danych.
- Parametry 𝐶 i jądro wymagają dostrojenia.
