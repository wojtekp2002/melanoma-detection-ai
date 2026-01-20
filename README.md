# Sprawozdanie projektowe  
## Aplikacja mobilna z wykorzystaniem sztucznej inteligencji do wykrywania czerniaka skóry

---

## 1. Wprowadzenie

Celem projektu było zaprojektowanie, implementacja oraz ewaluacja systemu wykorzystującego metody uczenia maszynowego do wspomagania wczesnego wykrywania czerniaka skóry na podstawie obrazu zmiany skórnej. System ma charakter **przesiewowy (screeningowy)** i nie stanowi diagnozy medycznej – jego zadaniem jest wsparcie użytkownika w podjęciu decyzji o ewentualnej konsultacji dermatologicznej.

Projekt obejmuje kompletny pipeline:
- przygotowanie danych,
- trenowanie i ewaluację modelu głębokiego uczenia,
- analizę metryk i dobór progu decyzyjnego,
- implementację backendowego API,
- przygotowanie systemu pod integrację z aplikacją mobilną.

---

## 2. Dane

### 2.1 Zbiór danych

W projekcie wykorzystano publiczny zbiór **HAM10000 (Human Against Machine with 10000 training images)**, zawierający 10 015 obrazów zmian skórnych pochodzących głównie z dermatoskopii oraz fotografii klinicznej.

Każdy obraz posiada etykietę diagnostyczną `dx`, którą sprowadzono do zadania binarnego:
- `mel` – czerniak (1),
- pozostałe klasy – zmiany łagodne (0).

Zbiór charakteryzuje się istotną nierównowagą klas – czerniaki stanowią ok. 11% próbek.

### 2.2 Podział danych

Dane zostały podzielone losowo w sposób stratyfikowany:
- **train** – 80% (uczenie modelu),
- **validation** – 10% (dobór hiperparametrów i progu),
- **test** – 10% (uczciwa ocena końcowa).

Podział ten zapobiega przeciekowi informacji i pozwala rzetelnie ocenić zdolność generalizacji modelu.

---

## 3. Model

### 3.1 Architektura

Zastosowano architekturę **EfficientNet-B0**, będącą nowoczesną konwolucyjną siecią neuronową zapewniającą dobry kompromis pomiędzy jakością predykcji a złożonością obliczeniową.

Wykorzystano **transfer learning**:
- załadowano wagi wstępnie wytrenowane na zbiorze ImageNet,
- zastąpiono końcową warstwę klasyfikującą pojedynczym neuronem z funkcją aktywacji sigmoid.

Model zwraca **prawdopodobieństwo wystąpienia czerniaka** w zakresie ⟨0, 1⟩.

### 3.2 Funkcja straty i optymalizacja

- Funkcja straty: `BCEWithLogitsLoss` z parametrem `pos_weight`, kompensującym nierównowagę klas.
- Optymalizator: `AdamW`.
- Scheduler: `ReduceLROnPlateau`.
- Zastosowano mechanizm **early stopping**, aby zapobiec przeuczeniu.

---

## 4. Metryki ewaluacyjne

### 4.1 AUC (Area Under ROC Curve)

AUC mierzy zdolność modelu do rozróżniania klas niezależnie od progu decyzyjnego:
- AUC = 0.5 – klasyfikacja losowa,
- AUC ≥ 0.9 – bardzo dobra jakość modelu.

### 4.2 Recall (Sensitivity, czułość)

Recall określa, jaki procent rzeczywistych czerniaków został poprawnie wykryty:
**Recall = TP / (TP + FN)**


W kontekście medycznym jest to **najważniejsza metryka**, ponieważ minimalizuje liczbę przeoczonych przypadków (FN).

### 4.3 Specificity (swoistość)

Specificity określa zdolność modelu do poprawnego rozpoznawania zmian łagodnych:
**Specificity = TN / (TN + FP)**


### 4.4 Precision

Precision informuje, jak często alarm wygenerowany przez model jest poprawny:
**Precision = TP / (TP + FP)**


---

## 5. Próg decyzyjny

Model zwraca prawdopodobieństwo, które musi zostać przekształcone w decyzję binarną przy pomocy **progu decyzyjnego (threshold)**.

Przeanalizowano kilka progów:
- 0.361 – maksymalizacja recall,
- 0.50 – próg domyślny,
- 0.559 – maksymalizacja swoistości.

Na podstawie wyników na zbiorze testowym wybrano **próg 0.50** jako najlepszy kompromis między czułością a swoistością oraz najbardziej stabilny pod względem generalizacji.

---

## 6. Eksperyment z augmentacjami „telefonicznymi”

Przeprowadzono eksperyment z augmentacjami symulującymi zdjęcia wykonywane smartfonem (rozmycie, zmiany oświetlenia, losowe kadrowanie).

### Wyniki eksperymentu:
- spadek wartości AUC,
- spadek swoistości,
- niewielki wzrost recall kosztem dużej liczby fałszywych alarmów.

Wniosek: augmentacje te nie poprawiły ogólnej jakości modelu dla dostępnych danych, dlatego **nie zostały użyte w wersji końcowej**.

---

## 7. Wyniki końcowe (zbiór testowy)

Dla wybranego modelu i progu 0.50 uzyskano:
- **AUC ≈ 0.94**,
- **Recall ≈ 0.86**,
- **Specificity ≈ 0.89**.

Wyniki te potwierdzają dobrą zdolność generalizacji modelu.

---

## 8. Backend API

Zaimplementowano backend w technologii **FastAPI**, udostępniający:
- `GET /health` – sprawdzenie stanu serwera,
- `POST /predict` – przesyłanie obrazu i zwrot wyniku klasyfikacji.

API:
- wykonuje preprocessing zgodny z treningiem,
- wykorzystuje GPU (CUDA), jeśli jest dostępne,
- zwraca prawdopodobieństwo oraz etykietę `low_risk` / `high_risk`,
- zawiera wyraźny komunikat o niemedycznym charakterze wyniku.

---

## 9. Ograniczenia i aspekty etyczne

- Model został wytrenowany głównie na danych dermatoskopowych, możliwe jest wystąpienie przesunięcia dziedziny (domain shift) dla zdjęć z telefonu.
- System nie zastępuje lekarza i pełni wyłącznie rolę wspomagającą.
- W aplikacji przewidziano jasne komunikaty ostrzegawcze dla użytkownika.

---

## 10. Instrukcja uruchomienia i użytkowania API

### 10.1 Co jest potrzebne do uruchomienia API

Wymagane:
- Python 3.11,
- plik z wytrenowanym modelem:
    **artifacts/efficientnet_b0_best.pt**
- pliki źródłowe API,
- zainstalowane zależności (`requirements.txt`).


### 10.2 Instalacja zależności

Po sklonowaniu repozytorium należy zainstalować wymagane biblioteki:

```bash
pip install -r requirements.txt
```

## 10.3 Uruchomienie serwera API

Backend aplikacji został zaimplementowany z wykorzystaniem frameworka **FastAPI**.
Aby uruchomić serwer API lokalnie, należy w katalogu głównym projektu wykonać polecenie:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```







