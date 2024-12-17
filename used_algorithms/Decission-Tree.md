# Decision Tree

Rozumiem, zajmijmy się implementacją algorytmu Decision Tree w funkcji decision_tree_classifier. Wyjaśnię, jak ten algorytm działa, a następnie pokażę, jak go zaimplementować w Pythonie, wykorzystując bibliotekę Scikit-learn.

## Jak działa Decision Tree?

Decision Tree to algorytm klasyfikacji, który działa jak drzewo decyzyjne. Na każdym węźle dokonuje podziału danych na podstawie warunku (np. czy wartość cechy x jest większa od pewnej wartości). Algorytm uczy się tych warunków na podstawie danych treningowych, aby podzielić dane na kategorie w najbardziej efektywny sposób.

Działa dobrze na małych i średnich zestawach danych, jest łatwy do interpretacji, ale może być podatny na overfitting (przeuczenie) przy braku ograniczenia głębokości drzewa.

## Zalety i Wady