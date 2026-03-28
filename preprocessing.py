import os
import re
import pandas as pd
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


def run_quota_preprocessing(app_ids_list, input_folder, output_folder, target_per_language=3000):
    print(f"=== ROZPOCZĘCIE ZBIERANIA DANYCH (CEL: {target_per_language} recenzji na język) ===")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    all_files = os.listdir(input_folder)

    # Koszyki na recenzje
    pl_reviews = pd.DataFrame()
    en_reviews = pd.DataFrame()

    for app_id in app_ids_list:
        # Sprawdzamy, czy mamy już wystarczająco dużo danych
        if len(pl_reviews) >= target_per_language and len(en_reviews) >= target_per_language:
            print("\n[SUKCES] Osiągnięto wymagany limit dla obu języków! Zatrzymuję przeszukiwanie.")
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

            # Aby nie wykrywać języka dla miliona wierszy, bierzemy próbkę np. 10 000
            n_samples = min(10000, len(df))
            df_sampled = df.sample(n=n_samples, random_state=42).copy()

            print(f" -> Detekcja języka ({n_samples} próbek)...")
            df_sampled['language'] = df_sampled['review'].apply(get_language)

            # Dodajemy znalezione polskie recenzje do koszyka
            new_pl = df_sampled[df_sampled['language'] == 'pl']
            pl_reviews = pd.concat([pl_reviews, new_pl])

            # Dodajemy znalezione angielskie recenzje do koszyka
            new_en = df_sampled[df_sampled['language'] == 'en']
            en_reviews = pd.concat([en_reviews, new_en])

            print(
                f" -> Stan koszyków: PL: {len(pl_reviews)}/{target_per_language} | EN: {len(en_reviews)}/{target_per_language}")

        except Exception as e:
            print(f"[BŁĄD] {e}")

    # --- BALANSOWANIE I CZYSZCZENIE ---
    print("\n=== Przycinanie, Balansowanie i Czyszczenie ===")
    final_dfs = []

    for lang, df_lang in [('pl', pl_reviews), ('en', en_reviews)]:
        if len(df_lang) < target_per_language:
            print(
                f"[OSTRZEŻENIE] Zebrano tylko {len(df_lang)} recenzji dla języka {lang}. To może być za mało dla Bi-LSTM!")

        # Przycinamy nadmiar, żeby było równo
        df_lang = df_lang.head(target_per_language)

        # Balansujemy klasy w obrębie języka (np. 1500 pozytywów, 1500 negatywów)
        positives = df_lang[df_lang['label'] == 1]
        negatives = df_lang[df_lang['label'] == 0]

        min_class_count = min(len(positives), len(negatives))
        print(f"[{lang.upper()}] Zbalansowano do {min_class_count} pozytywnych i {min_class_count} negatywnych.")

        balanced_df = pd.concat([
            positives.sample(n=min_class_count, random_state=42),
            negatives.sample(n=min_class_count, random_state=42)
        ])

        balanced_df['cleaned_review'] = balanced_df['review'].apply(clean_text)
        balanced_df = balanced_df[balanced_df['cleaned_review'] != '']
        final_dfs.append(balanced_df)

    # --- ZAPIS ---
    for df_lang in final_dfs:
        if df_lang.empty: continue
        lang = df_lang['language'].iloc[0]

        X_train, X_test, y_train, y_test = train_test_split(
            df_lang['cleaned_review'], df_lang['label'], test_size=0.2, random_state=42, stratify=df_lang['label']
        )

        pd.DataFrame({'review': X_train, 'cleaned_review': X_train, 'label': y_train}).to_csv(
            f"{output_folder}/train_{lang}.csv", index=False)
        pd.DataFrame({'review': X_test, 'cleaned_review': X_test, 'label': y_test}).to_csv(
            f"{output_folder}/test_{lang}.csv", index=False)

        print(f"Zapisano gotowy zbiór {lang.upper()}: Train={len(X_train)}, Test={len(X_test)}")