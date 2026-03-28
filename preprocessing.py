import os
import re
import pandas as pd
import numpy as np
from langdetect import detect, DetectorFactory
from sklearn.model_selection import train_test_split

# ==============================================================================
# KONFIGURACJA POCZĄTKOWA
# ==============================================================================
# Zapewnia powtarzalność wyników detektora języka (kluczowe dla badań naukowych)
DetectorFactory.seed = 42


# ==============================================================================
# FUNKCJE POMOCNICZE
# ==============================================================================

def get_language(text):
    """
    Bezpiecznie wykrywa język tekstu.
    Zwraca kod ISO 639-1 (np. 'en', 'pl') lub 'unknown' w przypadku błędu.
    """
    try:
        return detect(str(text))
    except:
        return 'unknown'


def clean_text(text):
    """
    Wykonuje podstawowe czyszczenie (preprocessing) tekstu recenzji.
    - Zamienia wszystkie litery na małe.
    - Usuwa tagi HTML.
    - Usuwa interpunkcję i znaki specjalne (zostawia tylko alfanumeryczne i spacje).
    - Usuwa wielokrotne spacje.
    """
    text = str(text).lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def balance_dataset(df):
    """
    Wykonuje under-sampling, aby zbiór był idealnie zbalansowany.
    Dla każdego języka w DataFrame wyrównuje liczbę recenzji pozytywnych (1)
    do liczby recenzji negatywnych (0), losując odpowiednią ilość próbek z klasy większościowej.
    """
    balanced_dfs = []

    for lang in df['language'].unique():
        df_lang = df[df['language'] == lang]

        positives = df_lang[df_lang['label'] == 1]
        negatives = df_lang[df_lang['label'] == 0]

        # Znajdujemy, której klasy jest mniej (wąskie gardło)
        min_count = min(len(positives), len(negatives))
        print(f"[{lang}] Balansowanie klas: pozostawiono {min_count} recenzji pozytywnych i {min_count} negatywnych.")

        if min_count == 0:
            print(f"[{lang}] UWAGA: Brak danych dla jednej z klas. Pomijanie języka.")
            continue

        # Losujemy próbki z obu klas, aby miały równą liczebność
        pos_sampled = positives.sample(n=min_count, random_state=42)
        neg_sampled = negatives.sample(n=min_count, random_state=42)

        balanced_dfs.append(pd.concat([pos_sampled, neg_sampled]))

    if balanced_dfs:
        return pd.concat(balanced_dfs, ignore_index=True)
    else:
        return pd.DataFrame()


# ==============================================================================
# GŁÓWNA LOGIKA (PIPELINE)
# ==============================================================================

def run_preprocessing_pipeline(app_ids_list, input_folder, output_folder, sample_per_game=10000):
    """
    Główna funkcja orkiestrująca cały proces przygotowania danych.
    1. Przeszukuje 'input_folder' w poszukiwaniu plików po AppID.
    2. Pobiera próbkę recenzji, wykrywa język i filtruje EN/PL.
    3. Czyści tekst i mapuje etykiety tekstowe na 1/0.
    4. Balansuje klasy, dzieli na Train/Test i zapisuje do 'output_folder'.
    """
    print(f"=== ROZPOCZĘCIE PREPROCESSINGU DLA {len(app_ids_list)} GIER ===")

    # Tworzenie folderu wyjściowego, jeśli nie istnieje
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    all_reviews = []

    # Pobieramy listę wszystkich plików w folderze wejściowym
    try:
        all_files_in_folder = os.listdir(input_folder)
    except FileNotFoundError:
        print(f"[BŁĄD KRYTYCZNY] Folder wejściowy '{input_folder}' nie istnieje!")
        return

    # KROK 1: EKSTRAKCJA I DETEKCJA JĘZYKA
    for app_id in app_ids_list:
        # Szukamy pliku zaczynającego się od AppID i podkreślenia (np. "730_154320.csv")
        matching_files = [f for f in all_files_in_folder if f.startswith(f"{app_id}_") and f.endswith('.csv')]

        if not matching_files:
            print(f"[UWAGA] Pominięto AppID: {app_id} - nie znaleziono pasującego pliku CSV.")
            continue

        game_file = matching_files[0]
        file_path = os.path.join(input_folder, game_file)

        print(f"\nPrzetwarzanie pliku: {game_file} (AppID: {app_id})...")

        try:
            # Wczytujemy plik CSV
            df = pd.read_csv(file_path)

            # Weryfikacja wymaganych kolumn
            if 'review' not in df.columns or 'recommend' not in df.columns:
                print(f"[BŁĄD] Plik {game_file} nie zawiera kolumn 'review' lub 'recommend'.")
                continue

            # Wybieramy tekst i ocenę, usuwamy puste wiersze
            df = df[['review', 'recommend']].dropna(subset=['review', 'recommend'])

            # Mapowanie etykiet: 'Recommended' -> 1, 'Not Recommended' -> 0
            label_mapping = {'Recommended': 1, 'Not Recommended': 0}
            df['label'] = df['recommend'].map(label_mapping)

            # Usuwamy wiersze, gdzie mapowanie się nie powiodło (powstały wartości NaN)
            df = df.dropna(subset=['label'])
            df['label'] = df['label'].astype(int)

            # Losowanie próby przed bardzo kosztowną czasowo detekcją języka
            n_samples = min(sample_per_game, len(df))
            df_sampled = df.sample(n=n_samples, random_state=42).copy()

            print(f" -> Detekcja języka dla próby {n_samples} recenzji (proszę czekać)...")
            df_sampled['language'] = df_sampled['review'].apply(get_language)

            # Zostawiamy wyłącznie recenzje angielskie ('en') i polskie ('pl')
            df_filtered = df_sampled[df_sampled['language'].isin(['en', 'pl'])]
            print(f" -> Wyodrębniono {len(df_filtered)} recenzji (EN/PL) dla tej gry.")

            all_reviews.append(df_filtered)

        except Exception as e:
            print(f"[BŁĄD] Wystąpił problem podczas przetwarzania pliku {game_file}: {e}")

    # Sprawdzenie, czy udało się cokolwiek przetworzyć
    if not all_reviews:
        print("\n[ZAKOŃCZONO] Brak poprawnych danych do dalszego przetwarzania.")
        return

    df_combined = pd.concat(all_reviews, ignore_index=True)
    print(f"\n=== Zakończono ekstrakcję. Łączna liczba recenzji (EN+PL): {len(df_combined)} ===")

    # KROK 2: CZYSZCZENIE TEKSTU (NLP)
    print("Rozpoczęto czyszczenie tekstu (usuwanie interpunkcji, formatowanie)...")
    df_combined['cleaned_review'] = df_combined['review'].apply(clean_text)

    # Usuwamy recenzje, które po wyczyszczeniu stały się puste (np. zawierały same emotikony)
    df_combined = df_combined[df_combined['cleaned_review'] != '']

    # KROK 3: BALANSOWANIE KLAS
    print("\nRozpoczęto balansowanie klas (Under-sampling)...")
    df_final = balance_dataset(df_combined)

    if df_final.empty:
        print("[BŁĄD] Zbiór danych po balansowaniu jest pusty.")
        return
    # KROK 4: PODZIAŁ NA ZBIORY (TRAIN/TEST) I ZAPIS
    print("\nRozpoczęto podział na zbiory treningowe (80%) i testowe (20%) oraz zapis...")
    for lang in ['en', 'pl']:
        df_lang = df_final[df_final['language'] == lang]

        if df_lang.empty:
            print(f"[UWAGA] Brak danych do zapisu dla języka: {lang}")
            continue

        # Dzielimy cały DataFrame (df_lang), a nie tylko pojedyncze kolumny
        df_train, df_test = train_test_split(
            df_lang,
            test_size=0.2,
            random_state=42,
            stratify=df_lang['label']
        )

        # Zapisujemy wybrane kolumny: oryginał, wyczyszczona i etykieta
        columns_to_save = ['review', 'cleaned_review', 'label']

        train_file = os.path.join(output_folder, f'train_{lang}.csv')
        test_file = os.path.join(output_folder, f'test_{lang}.csv')

        df_train[columns_to_save].to_csv(train_file, index=False)
        df_test[columns_to_save].to_csv(test_file, index=False)

        print(f"Zapisano dane dla [{lang.upper()}]: {len(df_train)} treningowych, {len(df_test)} testowych.")


    print("\n=== PREPROCESSING ZAKOŃCZONY SUKCESEM ===")
