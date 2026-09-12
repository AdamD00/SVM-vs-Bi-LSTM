import os
import pandas as pd
from datetime import datetime

def wygeneruj_porownanie(jezyk):
    print(f"Generowanie porównania dla języka: {jezyk.upper()}...")
    folder = './processed_data/'

    plik_svm = os.path.join(folder, f'predykcje_svm_{jezyk}.csv')
    plik_bilstm = os.path.join(folder, f'predykcje_bilstm_{jezyk}.csv')

    if not os.path.exists(plik_svm) or not os.path.exists(plik_bilstm):
        print(f"[BŁĄD] Brak plików z predykcjami dla języka {jezyk}.")
        return

    df_svm = pd.read_csv(plik_svm)
    df_bilstm = pd.read_csv(plik_bilstm)

    # Skoro zbiór testowy w obu modelach był podawany w tej samej kolejności,
    # możemy je połączyć w jedną tabelę
    df_zestawienie = pd.DataFrame({
        'Recenzja': df_svm['Wyczyszczona_Recenzja'],
        'Prawdziwa_Ocena': df_svm['Faktyczna_Ocena'],
        'Ocena_SVM': df_svm['Przewidywanie_SVM'],
        'Ocena_BiLSTM': df_bilstm['Przewidywanie_BiLSTM']
    })

    # Szukamy miejsc, gdzie modele się kłócą (jeden ma rację, a drugi się myli)
    # Warunek: Prawdziwa ocena zgadza się z SVM, a nie zgadza z Bi-LSTM... LUB ODWROTNIE
    df_roznice = df_zestawienie[df_zestawienie['Ocena_SVM'] != df_zestawienie['Ocena_BiLSTM']].copy()

    df_roznice['Zwyciezca'] = df_roznice.apply(kto_mial_racje, axis=1)

    # Pobieramy maksymalnie 100 losowych przypadków
    ilosc_do_pobrania = min(100, len(df_roznice))
    df_probka = df_roznice.sample(n=ilosc_do_pobrania, random_state=42)

    plik_wynikowy = os.path.join(folder, f'porownanie_100_roznic_{jezyk}.csv')
    df_probka.to_csv(plik_wynikowy, index=False, encoding='utf-8-sig')

    print(f" -> Znaleziono {len(df_roznice)} różnic w ocenach pomiędzy modelami.")
    print(f" -> Zapisano {ilosc_do_pobrania} przypadków do pliku: {plik_wynikowy}\n")

def kto_mial_racje(row):
    if row['Ocena_SVM'] == row['Prawdziwa_Ocena']:
        return "SVM"
    else:
        return "Bi-LSTM"

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
    PLIK_WYNIKOWY = f'Wyniki/Podsumowanie_{data_wykonania_nazwa_pliku}.txt'

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