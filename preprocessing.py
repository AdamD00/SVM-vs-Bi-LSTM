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
            print("\n[SUKCES] Zebrano idealnie zbalansowane dane dla obu języków!")
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