import os
import re
import random
import pandas as pd
import concurrent.futures
from langdetect import detect, DetectorFactory
from sklearn.model_selection import train_test_split

DetectorFactory.seed = 42


def get_language(text):
    try:
        return detect(str(text))
    except:
        return 'unknown'


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def process_single_file(file_path):
    """
    Funkcja "Workera" (Robotnika).
    Jeden rdzeń procesora bierze jeden plik CSV, wczytuje go, filtruje,
    wykrywa język i zwraca paczkę polskich i angielskich recenzji.
    """
    try:
        df = pd.read_csv(file_path)
        if 'review' not in df.columns or 'recommend' not in df.columns:
            return pd.DataFrame(), pd.DataFrame()  # Zwraca puste ramki, jeśli zły plik

        df = df[['review', 'recommend']].dropna()
        df['label'] = df['recommend'].map({'Recommended': 1, 'Not Recommended': 0})
        df = df.dropna(subset=['label'])
        df['label'] = df['label'].astype(int)

        # Bierzemy maks 3000 wierszy na plik dla szybkości
        n_samples = min(3000, len(df))
        df_sampled = df.sample(n=n_samples, random_state=42).copy()

        df_sampled['language'] = df_sampled['review'].apply(get_language)

        # Wyciągamy to co nas interesuje
        df_pl = df_sampled[df_sampled['language'] == 'pl']
        df_en = df_sampled[df_sampled['language'] == 'en']

        return df_pl, df_en

    except Exception as e:
        return pd.DataFrame(), pd.DataFrame()


def run_parallel_preprocessing(input_folder, output_folder, neg_pl_limit = 10000):
    print("=== ROZPOCZĘCIE ZBIERANIA DANYCH (MULTIPROCESSING) ===")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    all_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
    random.seed(42)
    random.shuffle(all_files)

    # Tworzymy 4 główne koszyki
    pl_pos = pd.DataFrame()
    pl_neg = pd.DataFrame()
    en_pos = pd.DataFrame()
    en_neg = pd.DataFrame()

    # Obliczamy ile mamy rdzeni procesora (np. 8 lub 16)
    max_workers = os.cpu_count() or 4
    print(f"Uruchamianie na {max_workers} rdzeniach procesora równocześnie!")

    przeszukane_gry = 0

    # Uruchamiamy Pule Procesów
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        przypisane_zadania = {executor.submit(process_single_file, os.path.join(input_folder, f)): f for f in all_files}
        for future in concurrent.futures.as_completed(przypisane_zadania):
            plik_csv = przypisane_zadania[future]
            przeszukane_gry += 1

            try:
                # Odbieramy wyniki od robotnika
                df_pl, df_en = future.result()

                # Uzupełniamy główne koszyki
                if not df_pl.empty:
                    pl_pos = pd.concat([pl_pos, df_pl[df_pl['label'] == 1]])
                    pl_neg = pd.concat([pl_neg, df_pl[df_pl['label'] == 0]])
                if not df_en.empty:
                    en_pos = pd.concat([en_pos, df_en[df_en['label'] == 1]])
                    en_neg = pd.concat([en_neg, df_en[df_en['label'] == 0]])

            except Exception as e:
                print(f"Błąd przy przetwarzaniu {plik_csv}: {e}")

            # Wyświetlanie statusu
            if przeszukane_gry % 10 == 0:
                print(f"Przetworzono {przeszukane_gry} plików. Stan koszyków:")
                print(f" -> PL: (Poz: {len(pl_pos)}, Neg: {len(pl_neg)})")
                print(f" -> EN: (Poz: {len(en_pos)}, Neg: {len(en_neg)})")

            if len(pl_neg) > neg_pl_limit:
                print(f"\n[SUKCES] Zebrano ponad {neg_pl_limit} trudnych recenzji! Zatrzymuję pozostałe rdzenie...")
                # Anulujemy te pliki, które jeszcze nie zdążyły się uruchomić
                for task in przypisane_zadania:
                    task.cancel()
                break

    # --- BALANSOWANIE I CZYSZCZENIE ---
    print("\n=== Skanowanie zakończone. Balansowanie do najsłabszej klasy ===")

    min_count = min(len(pl_pos), len(pl_neg), len(en_pos), len(en_neg))
    print(f"Najmniej liczna klasa posiada {min_count} próbek. Równanie zbiorów...")

    pl_pos = pl_pos.sample(n=min_count, random_state=42)
    pl_neg = pl_neg.sample(n=min_count, random_state=42)
    en_pos = en_pos.sample(n=min_count, random_state=42)
    en_neg = en_neg.sample(n=min_count, random_state=42)

    df_pl = pd.concat([pl_pos, pl_neg])
    df_en = pd.concat([en_pos, en_neg])

    for df_lang, lang in [(df_pl, 'pl'), (df_en, 'en')]:
        if len(df_lang) == 0: continue

        print(f"Czyszczenie tekstu (NLP) dla {lang.upper()}...")
        df_lang['cleaned_review'] = df_lang['review'].apply(clean_text)
        df_lang = df_lang[df_lang['cleaned_review'] != '']

        X_train, X_test, y_train, y_test = train_test_split(
            df_lang['cleaned_review'], df_lang['label'], test_size=0.2, random_state=42, stratify=df_lang['label']
        )

        pd.DataFrame({'review': X_train, 'cleaned_review': X_train, 'label': y_train}).to_csv(
            f"{output_folder}/train_{lang}.csv", index=False)
        pd.DataFrame({'review': X_test, 'cleaned_review': X_test, 'label': y_test}).to_csv(
            f"{output_folder}/test_{lang}.csv", index=False)
        print(f"Zapisano gotowy zestaw {lang.upper()}: Train={len(X_train)}, Test={len(X_test)}")

def run_max_balanced_preprocessing(intput_folder,output_folder, review_number = 3000, neg_pl_limit = 10000):
    print("=== ROZPOCZĘCIE GLOBALNEGO ZBIERANIA DANYCH (MAX CAP) ===")
    print("Skrypt przeszuka próbki ze WSZYSTKICH plików i wyrówna zbiory do najsłabszej klasy.")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    all_files = [f for f in os.listdir(intput_folder) if f.endswith('.csv')]
    random.seed(42)
    random.shuffle(all_files)

    print(f"Znaleziono {len(all_files)} plików CSV z opiniami")
    pl_pos = pd.DataFrame()
    pl_neg = pd.DataFrame()
    en_pos = pd.DataFrame()
    en_neg = pd.DataFrame()

    przeszukane_gry = 0

    for game_file in all_files:
        file_path = os.path.join(intput_folder, game_file)
        przeszukane_gry += 1

        try:
            df = pd.read_csv(file_path)
            if 'review' not in df.columns or 'recommend' not in df.columns:
                continue

            df = df[['review', 'recommend']].dropna()
            df['label'] = df['recommend'].map({'Recommended': 1, 'Not Recommended': 0})
            df = df.dropna(subset=['label'])
            df['label'] = df['label'].astype(int)

            n_samples = min(review_number, len(df))
            df_sampled = df.sample(n=n_samples, random_state=42).copy()
            df_sampled['language'] = df_sampled['review'].apply(get_language)


            pl_pos = pd.concat([pl_pos, df_sampled[(df_sampled['language'] == 'pl') & (df_sampled['label'] == 1)]])
            pl_neg = pd.concat([pl_neg, df_sampled[(df_sampled['language'] == 'pl') & (df_sampled['label'] == 0)]])
            en_pos = pd.concat([en_pos, df_sampled[(df_sampled['language'] == 'en') & (df_sampled['label'] == 1)]])
            en_neg = pd.concat([en_neg, df_sampled[(df_sampled['language'] == 'en') & (df_sampled['label'] == 0)]])

            if przeszukane_gry % 5 == 0:
                print(f"Przeszukano gier: {przeszukane_gry}/{len(all_files)}. Obecny stan:")
                print(f" -> PL: (Poz: {len(pl_pos)}, Neg: {len(pl_neg)})")
                print(f" -> EN: (Poz: {len(en_pos)}, Neg: {len(en_neg)})")
            if len(pl_neg) > neg_pl_limit:
                print(f"Osiągnięto bazę >{neg_pl_limit} negatywnych recenzji. Kończę skanowanie!")
                break

        except Exception as e:
            continue
    # --- BALANSOWANIE DO NAJSŁABSZEGO OGNIWA ---
    print("\n=== Koniec skanowania. Równanie zbiorów do najsłabszej klasy ===")

    min_count = min(len(pl_pos), len(pl_neg), len(en_pos), len(en_neg))
    print(f"Najmniej liczna klasa posiada {min_count} próbek. Do tej wartości wyrównujemy wszystko.")

    pl_pos = pl_pos.sample(n=min_count, random_state=42)
    pl_neg = pl_neg.sample(n=min_count, random_state=42)
    en_pos = en_pos.sample(n=min_count, random_state=42)
    en_neg = en_neg.sample(n=min_count, random_state=42)

    df_pl = pd.concat([pl_pos, pl_neg])
    df_en = pd.concat([en_pos, en_neg])

    for df_lang, lang in [(df_pl, 'pl'), (df_en, 'en')]:
        if len(df_lang) == 0: continue

        df_lang['cleaned_review'] = df_lang['review'].apply(clean_text)
        df_lang = df_lang[df_lang['cleaned_review'] != '']

        X_train, X_test, y_train, y_test = train_test_split(
            df_lang['cleaned_review'], df_lang['label'], test_size=0.2, random_state=42, stratify=df_lang['label']
        )

        pd.DataFrame({'review': X_train, 'cleaned_review': X_train, 'label': y_train}).to_csv(
            f"{output_folder}/train_{lang}.csv", index=False)
        pd.DataFrame({'review': X_test, 'cleaned_review': X_test, 'label': y_test}).to_csv(
            f"{output_folder}/test_{lang}.csv", index=False)
        print(f"Zapisano {lang.upper()}: Train={len(X_train)}, Test={len(X_test)} (Idealnie zbalansowane!)")

def run_quota_preprocessing(app_ids_list, input_folder, output_folder, target_per_class=1500):
    print(
        f"=== ROZPOCZĘCIE ZBIERANIA DANYCH (CEL: {target_per_class} pozytywów i {target_per_class} negatywów na język) ===")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    all_files = os.listdir(input_folder)


    pl_pos = pd.DataFrame()
    pl_neg = pd.DataFrame()
    en_pos = pd.DataFrame()
    en_neg = pd.DataFrame()

    for app_id in app_ids_list:
        if len(pl_pos) >= target_per_class and len(pl_neg) >= target_per_class and \
                len(en_pos) >= target_per_class and len(en_neg) >= target_per_class:
            print("\n[SUKCES] Zebrano zbalansowane dane dla obu języków!")
            break

        matching_files = [f for f in all_files if f.startswith(f"{app_id}_") and f.endswith('.csv')]
        if not matching_files:
            continue

        game_file = matching_files[0]
        file_path = os.path.join(input_folder, game_file)
        print(f"\nSkanowanie pliku: {game_file}...")

        try:
            df = pd.read_csv(file_path)
            if 'review' not in df.columns or 'recommend' not in df.columns:
                continue

            df = df[['review', 'recommend']].dropna()
            df['label'] = df['recommend'].map({'Recommended': 1, 'Not Recommended': 0})
            df = df.dropna(subset=['label'])
            df['label'] = df['label'].astype(int)

            # Bierzemy próbkę recenzji do detekcji
            n_samples = min(20000, len(df))
            df_sampled = df.sample(n=n_samples, random_state=42).copy()

            df_sampled['language'] = df_sampled['review'].apply(get_language)
            # Segregowanie do odpowiednich koszyków
            if (len(pl_pos)/target_per_class) <1:
                pl_pos = pd.concat([pl_pos, df_sampled[(df_sampled['language'] == 'pl') & (df_sampled['label'] == 1)]])
                str_pl_pos = f"{(len(pl_pos)/target_per_class)}"
            else:
                str_pl_pos = 'FULL'
            if (len(pl_neg) / target_per_class) < 1:
                pl_neg = pd.concat([pl_neg, df_sampled[(df_sampled['language'] == 'pl') & (df_sampled['label'] == 0)]])
                str_pl_neg = f"{(len(pl_neg)/target_per_class)}"
            else:
                str_pl_neg = 'FULL'
            if (len(en_pos) / target_per_class) < 1:
                en_pos = pd.concat([en_pos, df_sampled[(df_sampled['language'] == 'en') & (df_sampled['label'] == 1)]])
                str_en_pos = f"{(len(en_pos)/target_per_class)}"
            else:
                str_en_pos = 'FULL'
            if (len(en_neg) / target_per_class) < 1:
                en_neg = pd.concat([en_neg, df_sampled[(df_sampled['language'] == 'en') & (df_sampled['label'] == 0)]])
                str_en_neg = f"{len(en_neg)}/{target_per_class}"
            else:
                str_en_neg = 'FULL'

            print(
                f" -> Zapełnienie PL: (Pozytywy: {str_pl_pos}, Negatywy: {str_pl_neg})")
            print(
                f" -> Zapełnienie EN: (Pozytywy: {str_en_pos}, Negatywy: {str_en_neg})")

        except Exception as e:
            print(f"[BŁĄD] {e}")
    #if not(len(pl_pos) >= target_per_class and len(pl_neg) >= target_per_class and len(en_pos) >= target_per_class and len(en_neg) >= target_per_class):

    # --- CZYSZCZENIE I ŁĄCZENIE KOSZYKÓW ---
    print("\n=== Czyszczenie i Zapis ===")

    # Przycinamy każdy koszyk dokładnie do limitu (np. 1500), żeby pozbyć się nadwyżek
    pl_pos = pl_pos.head(target_per_class)
    pl_neg = pl_neg.head(target_per_class)
    en_pos = en_pos.head(target_per_class)
    en_neg = en_neg.head(target_per_class)

    df_pl = pd.concat([pl_pos, pl_neg])
    df_en = pd.concat([en_pos, en_neg])

    for df_lang, lang in [(df_pl, 'pl'), (df_en, 'en')]:
        if len(df_lang) == 0: continue

        df_lang['cleaned_review'] = df_lang['review'].apply(clean_text)
        df_lang = df_lang[df_lang['cleaned_review'] != '']

        X_train, X_test, y_train, y_test = train_test_split(
            df_lang['cleaned_review'], df_lang['label'], test_size=0.2, random_state=42, stratify=df_lang['label']
        )

        pd.DataFrame({'review': X_train, 'cleaned_review': X_train, 'label': y_train}).to_csv(
            f"{output_folder}/train_{lang}.csv", index=False)
        pd.DataFrame({'review': X_test, 'cleaned_review': X_test, 'label': y_test}).to_csv(
            f"{output_folder}/test_{lang}.csv", index=False)
        print(f"Zapisano {lang.upper()}: łącznie {len(df_lang)} idealnie zbalansowanych recenzji.")