**Karta Tematu Pracy Licencjackiej** 

### **Temat (PL)**

**Prognozowanie zmienności i ryzyka Value-at-Risk z wykorzystaniem modeli rodziny GARCH oraz głębokich sieci rekurencyjnych: analiza porównawcza na rynkach akcyjnych**

### **Temat (ENG)**

**Volatility and Value-at-Risk Forecasting Using GARCH Family Models and Deep Recurrent Neural Networks: A Comparative Study on Equity Markets**

---

### **1\. Cel pracy**

Celem pracy jest porównanie skuteczności prognostycznej klasycznych modeli ekonometrycznych (z rodziny GARCH) oraz nowoczesnych metod uczenia maszynowego (Głębokie Sieci Rekurencyjne: LSTM i GRU) w zadaniu estymacji zmienności warunkowej oraz ryzyka rynkowego (Value-at-Risk). Badanie ma na celu sprawdzenie, czy modele oparte na sztucznej inteligencji oferują istotną statystycznie przewagę nad modelami parametrycznymi w różnych warunkach rynkowych.

### **2\. Hipotezy badawcze**

1. **H1:** Modele oparte na głębokim uczeniu (LSTM/GRU) osiągają mniejsze błędy prognoz (RMSE, QLIKE) niż klasyczne modele GARCH w okresach podwyższonej zmienności rynkowej.  
2. **H2:** Uwzględnienie efektu asymetrii (modele EGARCH/GJR-GARCH) pozwala na skuteczniejszą estymację ryzyka (VaR) niż podstawowy model GARCH(1,1) oraz modele sieciowe.  
3. **H3:** Złożone modele sieci neuronowych są bardziej podatne na przeuczenie (overfitting) przy ograniczonej próbie danych dziennych w porównaniu do modeli ekonometrycznych.

---

### **3\. Dane i Zakres Badania**

**Okres badawczy:**

* Styczeń 2005 – Grudzień 2025 (15 lat).  
* Częstotliwość: Dzienna (Daily Adjusted Close).  
* Dane: YahooFinance

**Instrumenty (Zróżnicowanie sektorowe i geograficzne):**

1. **Technologia (USA):** Apple (AAPL), NVIDIA (NVDA) – reprezentacja wysokiej zmienności (high-beta).  
2. **Finanse (USA):** JP Morgan (JPM) – reprezentacja sektora bankowego.  
3. **Energia (USA):** Exxon Mobil (XOM) – reprezentacja sektora surowcowego.  
4. **Rynek Polski:** Southern Copper corporation (SCCO) 

**Przygotowanie danych (Preprocessing):**

* Obliczenie logarytmicznych stóp zwrotu: rt​=ln(Pt​/Pt−1​).  
* Standaryzacja/Normalizacja danych (niezbędna dla sieci neuronowych).  
* Podział na zbiory: Treningowy (70%), Walidacyjny (15%), Testowy (15%) lub zastosowanie Walidacji Kroczącej (Walk-Forward).

---

### **4\. Metodologia**

#### **A. Modele Ekonometryczne (Benchmarki)**

1. **GARCH(1,1):** Model bazowy (Standard Benchmark).  
2. **EGARCH:** Do modelowania asymetrii informacji (reakcja na "dobre" i "złe" newsy).  
3. **GJR-GARCH:** Alternatywa dla asymetrii.

#### **B. Modele Deep Learning**

1. **LSTM (Long Short-Term Memory):** Sieć zdolna do wyłapywania długoterminowych zależności w szeregach czasowych.  
2. **GRU (Gated Recurrent Unit):** Uproszczona wersja LSTM, często szybsza w treningu i równie skuteczna przy mniejszych zbiorach danych.

#### **C. Metodyka Treningu i Walidacji**

* **Walk-Forward Validation:** Iteracyjne przesuwanie okna treningowego, aby symulować rzeczywisty proces inwestycyjny (uniknięcie *look-ahead bias*).  
* **Proxy Zmienności (Target):** Jako estymator "prawdziwej" zmienności (która jest nieobserwowalna) wykorzystane zostaną **kwadraty stóp zwrotu** (rt2​).

---

### **5\. Metryki Oceny i Testy Statystyczne**

**Oceny błędów prognozy (Forecast Accuracy):**

* **RMSE (Root Mean Squared Error):** Standardowa miara błędu.  
* **MAE (Mean Absolute Error):** Miara odporna na wartości odstające.  
* **QLIKE Loss:** Asymetryczna funkcja straty, standard w literaturze dotyczącej zmienności (karze niedoszacowanie zmienności bardziej niż przeszacowanie). DOCZYTAĆ  
* MAPE \-

**Porównanie Modeli:**

* **Test Diebolda-Mariano (DM Test):** Weryfikacja, czy różnice w wynikach modeli są istotne statystycznie.

---

### **7\. Przewidywane Wyzwania i Ryzyka**

1. **Zmienność jako zmienna ukryta:** Brak możliwości bezpośredniegow pomiaru "prawdziwej" zmienności utrudnia ocenę modeli (konieczność stosowania *noisy proxy* jak kwadraty zwrotów).  
2. **Przeuczenie sieci (Overfitting):** Dane finansowe mają niski stosunek sygnału do szumu (low signal-to-noise ratio). Sieci LSTM mogą "zapamiętać" szum zamiast uczyć się wzorców.  
3. **Niestabilność parametrów GARCH:** W okresach kryzysów (np. COVID-19 w 2020 r.) algorytmy optymalizacyjne mogą mieć trudności ze zbieżnością.  
4. **Czas obliczeń:** Procedura *Walk-Forward* dla sieci neuronowych jest bardzo kosztowna obliczeniowo.

### **8\. Dodatki:**

1. SHAP/Explainability dla LSTM/GRU

