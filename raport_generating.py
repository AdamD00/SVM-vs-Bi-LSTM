import os
import pandas as pd
from datetime import datetime


def policz_recenzje(katalog_danych, jezyk):
    """Zlicza recenzje z plików train i test dla danego języka."""
    sciezka_train = os.path.join(katalog_danych, f'train_{jezyk}.csv')
    sciezka_test = os.path.join(katalog_danych, f'test_{jezyk}.csv')

    suma = 0
    if os.path.exists(sciezka_train):
        suma += len(pd.read_csv(sciezka_train))
    if os.path.exists(sciezka_test):
        suma += len(pd.read_csv(sciezka_test))
    return suma


def pobierz_wyniki(sciezka_do_pliku, jezyk):
    """Odczytuje wyniki modelu z wygenerowanego pliku CSV i ładnie je formatuje."""
    if not os.path.exists(sciezka_do_pliku):
        return {k: '[BRAK PLIKU]' for k in ['acc', 'prec', 'rec', 'f1', 'czas_tren', 'czas_inf']}

    df = pd.read_csv(sciezka_do_pliku)
    wiersz = df[df['Język'] == jezyk]

    if wiersz.empty:
        return {k: '[BRAK DANYCH]' for k in ['acc', 'prec', 'rec', 'f1', 'czas_tren', 'czas_inf']}

    wynik = wiersz.iloc[0]
    return {
        'acc': f"{wynik['Accuracy'] * 100:.2f}%",
        'prec': f"{wynik['Precision'] * 100:.2f}%",
        'rec': f"{wynik['Recall'] * 100:.2f}%",
        'f1': f"{wynik['F1-Score'] * 100:.2f}%",
        'czas_tren': f"{wynik['Czas Treningu [s]']:.4f} s",
        'czas_inf': f"{wynik['Czas Inferencji [s]']:.4f} s"
    }


def generuj_raport():
    KATALOG_DANYCH = './processed_data/'
    data_wykonania = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data_wykonania_nazwa_pliku = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    PLIK_WYNIKOWY = f'Wyniki/Podsumowanie_Eksperymentu_{data_wykonania_nazwa_pliku}.txt'

    # Zliczanie danych
    liczba_pl = policz_recenzje(KATALOG_DANYCH, 'pl')
    liczba_en = policz_recenzje(KATALOG_DANYCH, 'en')
    liczba_lacznie = liczba_pl + liczba_en

    # AUTOMATYCZNE POBIERANIE WYNIKÓW!
    print("Trwa odczytywanie wyników z plików CSV...")
    svm_en = pobierz_wyniki(os.path.join(KATALOG_DANYCH, 'wyniki_svm.csv'), 'EN')
    svm_pl = pobierz_wyniki(os.path.join(KATALOG_DANYCH, 'wyniki_svm.csv'), 'PL')
    bilstm_en = pobierz_wyniki(os.path.join(KATALOG_DANYCH, 'wyniki_bilstm.csv'), 'EN')
    bilstm_pl = pobierz_wyniki(os.path.join(KATALOG_DANYCH, 'wyniki_bilstm.csv'), 'PL')



    raport = f"""=================================================================
RAPORT Z EKSPERYMENTU BADAWCZEGO - ANALIZA SENTYMENTU
Wygenerowano: {data_wykonania}
=================================================================

CZĘŚĆ 1: ZESTAWIENIE DANYCH (DATASET)
-----------------------------------------------------------------
Łączna liczba przetworzonych recenzji: {liczba_lacznie}

Podział na języki (idealnie zbalansowane 50/50 pozytywne/negatywne):
- Język Polski (PL): {liczba_pl} recenzji
- Język Angielski (EN): {liczba_en} recenzji


CZĘŚĆ 2: WYNIKI MODELU KLASYCZNEGO (SVM + TF-IDF)
-----------------------------------------------------------------
[Język Angielski]
- Dokładność (Accuracy): {svm_en['acc']}
- Precyzja (Precision):  {svm_en['prec']}
- Czułość (Recall):      {svm_en['rec']}
- Miara F1 (Macro):      {svm_en['f1']}
- Czas treningu:         {svm_en['czas_tren']}
- Czas inferencji:       {svm_en['czas_inf']}

[Język Polski]
- Dokładność (Accuracy): {svm_pl['acc']}
- Precyzja (Precision):  {svm_pl['prec']}
- Czułość (Recall):      {svm_pl['rec']}
- Miara F1 (Macro):      {svm_pl['f1']}
- Czas treningu:         {svm_pl['czas_tren']}
- Czas inferencji:       {svm_pl['czas_inf']}


CZĘŚĆ 3: WYNIKI SIECI NEURONOWEJ (Bi-LSTM + Word Embeddings)
-----------------------------------------------------------------
[Język Angielski]
- Dokładność (Accuracy): {bilstm_en['acc']}
- Precyzja (Precision):  {bilstm_en['prec']}
- Czułość (Recall):      {bilstm_en['rec']}
- Miara F1 (Macro):      {bilstm_en['f1']}
- Czas treningu:         {bilstm_en['czas_tren']}
- Czas inferencji:       {bilstm_en['czas_inf']}

[Język Polski]
- Dokładność (Accuracy): {bilstm_pl['acc']}
- Precyzja (Precision):  {bilstm_pl['prec']}
- Czułość (Recall):      {bilstm_pl['rec']}
- Miara F1 (Macro):      {bilstm_pl['f1']}
- Czas treningu:         {bilstm_pl['czas_tren']}
- Czas inferencji:       {bilstm_pl['czas_inf']}
================================================================="""

    with open(PLIK_WYNIKOWY, 'w', encoding='utf-8') as plik:
        plik.write(raport)

    print(f"SUKCES! Raport został automatycznie wygenerowany do pliku: {PLIK_WYNIKOWY}")


if __name__ == "__main__":
    generuj_raport()