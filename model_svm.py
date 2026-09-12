import pandas as pd
import time
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, f1_score


# ==============================================================================
# FUNKCJA GŁÓWNA TRENINGU I EWALUACJI
# ==============================================================================

def train_and_evaluate_svm(language, data_folder):
    """
    Wczytuje przygotowane dane dla konkretnego języka, wektoryzuje je (TF-IDF),
    trenuje model SVM, przeprowadza testy i zwraca wyniki oraz czasy.
    """
    print(f"\n{'=' * 50}")
    print(f"URUCHAMIANIE MODELU SVM DLA JĘZYKA: [{language.upper()}]")
    print(f"{'=' * 50}")

    # 1. Wczytywanie danych
    train_path = os.path.join(data_folder, f'train_{language}.csv')
    test_path = os.path.join(data_folder, f'test_{language}.csv')

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"[BŁĄD] Brak plików z danymi dla języka '{language}'. Uruchom najpierw preprocessing.py")
        return

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    # Zabezpieczenie przed pustymi wartościami (NaN), które mogły powstać po czyszczeniu
    df_train['cleaned_review'] = df_train['cleaned_review'].fillna('')
    df_test['cleaned_review'] = df_test['cleaned_review'].fillna('')

    X_train_text = df_train['cleaned_review']
    y_train = df_train['label']

    X_test_text = df_test['cleaned_review']
    y_test = df_test['label']

    print(f"Wczytano danych treningowych: {len(X_train_text)}")
    print(f"Wczytano danych testowych: {len(X_test_text)}")

    # 2. Wektoryzacja TF-IDF
    print("\nTrwa wektoryzacja TF-IDF (zamiana tekstu na liczby)...")
    # max_features ogranicza słownik do 10 000 najważniejszych słów
    tfidf = TfidfVectorizer(max_features=10000)

    X_train_vec = tfidf.fit_transform(X_train_text)

    X_test_vec = tfidf.transform(X_test_text)

    # 3. Trening modelu SVM
    print("Trwa trening modelu SVM...")
    svm_model = SVC(kernel='linear', random_state=42)

    start_train_time = time.time()
    svm_model.fit(X_train_vec, y_train)
    end_train_time = time.time()

    train_time = end_train_time - start_train_time
    print(f" -> Czas treningu: {train_time:.4f} sekund")

    # 4. Inferencja (Testowanie)
    print("\nTrwa inferencja (przewidywanie na zbiorze testowym)...")

    start_infer_time = time.time()
    y_pred = svm_model.predict(X_test_vec)
    end_infer_time = time.time()
    # Tworzymy DataFrame z wynikami
    df_analysis = pd.DataFrame({
        'Wyczyszczona_Recenzja': X_test_text,
        'Faktyczna_Ocena': y_test,
        'Przewidywanie_SVM': y_pred
    })

    # Wyciągamy recenzje, w których model się POMYLIŁ
    df_errors = df_analysis[df_analysis['Faktyczna_Ocena'] != df_analysis['Przewidywanie_SVM']]

    # Wyciągamy recenzje, w których model miał RACJĘ
    df_success = df_analysis[df_analysis['Faktyczna_Ocena'] == df_analysis['Przewidywanie_SVM']]

    error_file = os.path.join(data_folder, f'svm_bledy_{language}.csv')
    success_file = os.path.join(data_folder, f'svm_sukcesy_{language}.csv')

    df_errors.sample(min(50, len(df_errors)), random_state=42).to_csv(error_file, index=False)
    df_success.sample(min(50, len(df_success)), random_state=42).to_csv(success_file, index=False)
    df_analysis.to_csv(os.path.join(data_folder, f'predykcje_svm_{language}.csv'), index=False)
    print(f"\nZapisano przykłady do analizy w folderze {data_folder}:")
    print(f" -> Zobacz plik: svm_bledy_{language}.csv (żeby zobaczyć, gdzie model poległ)")

    infer_time = end_infer_time - start_infer_time
    # Liczymy czas potrzebny na jedną recenzję (w milisekundach)
    infer_time_per_sample = (infer_time / len(y_test)) * 1000
    print(f" -> Całkowity czas inferencji: {infer_time:.4f} sekund")
    print(f" -> Średni czas na 1 recenzję: {infer_time_per_sample:.4f} ms")

    # 5. Wyniki i metryki
    print("\n--- RAPORT WYNIKÓW ---")
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')

    print(f"Accuracy (Dokładność): {acc:.4f}")
    print(f"F1-Score (Macro):      {f1:.4f}")
    print("\nSzczegółowy raport klasyfikacji:")
    report_dict = classification_report(y_test, y_pred, output_dict=True)

    return {
        'Język': language.upper(),
        'Accuracy': acc,
        'Precision': report_dict['macro avg']['precision'],
        'Recall': report_dict['macro avg']['recall'],
        'F1-Score': f1,
        'Czas Treningu [s]': train_time,
        'Czas Inferencji [s]': infer_time
    }