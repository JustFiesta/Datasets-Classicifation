# Decision Tree

Rozumiem, zajmijmy się implementacją algorytmu Decision Tree w funkcji decision_tree_classifier. Wyjaśnię, jak ten algorytm działa, a następnie pokażę, jak go zaimplementować w Pythonie, wykorzystując bibliotekę Scikit-learn.

## Jak działa Decision Tree?

Decision Tree to algorytm klasyfikacji, który działa jak drzewo decyzyjne. Na każdym węźle dokonuje podziału danych na podstawie warunku (np. czy wartość cechy x jest większa od pewnej wartości). Algorytm uczy się tych warunków na podstawie danych treningowych, aby podzielić dane na kategorie w najbardziej efektywny sposób.

Działa dobrze na małych i średnich zestawach danych, jest łatwy do interpretacji, ale może być podatny na overfitting (przeuczenie) przy braku ograniczenia głębokości drzewa.

## Zalety i Wady

Zalety:

- Prostota i interpretowalność: Łatwe do zrozumienia i wizualizacji, co umożliwia interpretację procesu podejmowania decyzji.
- Brak potrzeby normalizacji danych: Nie wymaga skalowania zmiennych wejściowych.
- Radzenie sobie z danymi nieliniowymi: Może modelować nieliniowe zależności.
- Możliwość obsługi danych zarówno numerycznych, jak i kategorycznych.

Wady:

- Łatwość w przeuczeniu: Może łatwo dopasować się do danych treningowych, co prowadzi do overfittingu.
- Niestałość: Mała zmiana w danych treningowych może prowadzić do znacznych zmian w strukturze drzewa.
- Ograniczona dokładność: Przy skomplikowanych danych może nie być tak skuteczne jak inne algorytmy (np. Random Forest).