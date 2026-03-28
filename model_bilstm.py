import pandas as pd
import time
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense
from sklearn.metrics import classification_report, accuracy_score, f1_score

# ==============================================================================
# HIPERPARAMETRY SIECI
# ==============================================================================
MAX_WORDS = 10000  # Rozmiar słownika (tyle samo co w SVM dla uczciwego porównania)
MAX_SEQUENCE_LEN = 100  # Do ilu słów ucinamy/wydłużamy recenzję (tzw. padding)
EMBEDDING_DIM = 64  # Rozmiar wektora osadzeń (jak bardzo złożone jest "zrozumienie" słowa)
LSTM_UNITS = 32  # Liczba komórek pamięci w LSTM
EPOCHS = 5  # Ile razy model zobaczy cały zbiór danych podczas nauki
BATCH_SIZE = 64  # Ile recenzji na raz model pakuje do pamięci RAM/VRAM


# ==============================================================================
# GŁÓWNA LOGIKA
# ==============================================================================

def train_and_evaluate_bilstm(language, data_folder):
    print(f"\n{'=' * 50}")
    print(f"URUCHAMIANIE MODELU BI-LSTM DLA JĘZYKA: [{language.upper()}]")
    print(f"{'=' * 50}")

    # 1. Wczytywanie danych
    train_path = os.path.join(data_folder, f'train_{language}.csv')
    test_path = os.path.join(data_folder, f'test_{language}.csv')

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"[BŁĄD] Brak plików z danymi dla języka '{language}'.")
        return None

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    # Zabezpieczenie przed pustymi stringami
    X_train_text = df_train['cleaned_review'].fillna('').astype(str)
    y_train = df_train['label'].values

    X_test_text = df_test['cleaned_review'].fillna('').astype(str)
    y_test = df_test['label'].values

    # 2. Tokenizacja i Wyrównywanie sekwencji (Padding)
    print("\nTrwa przygotowywanie sekwencji słów dla sieci neuronowej...")
    tokenizer = Tokenizer(num_words=MAX_WORDS)
    tokenizer.fit_on_texts(X_train_text)  # Model tworzy sobie własny słownik na podstawie danych treningowych

    # Zamiana tekstu na ciągi cyfr (np. "dobra gra" -> [34, 12])
    X_train_seq = tokenizer.texts_to_sequences(X_train_text)
    X_test_seq = tokenizer.texts_to_sequences(X_test_text)

    # Padding: Sieć neuronowa wymaga wejścia o stałej długości.
    # Za krótkie recenzje dostaną zera z przodu, za długie zostaną obcięte.
    X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_SEQUENCE_LEN)
    X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_SEQUENCE_LEN)

    # 3. Budowa Architektury Modelu
    print("Budowanie architektury Bi-LSTM...")
    model = Sequential([
        # Warstwa Embedding: Zmienia cyfry w gęste wektory. To tutaj model uczy się znaczenia słów.
        Embedding(input_dim=MAX_WORDS, output_dim=EMBEDDING_DIM, input_length=MAX_SEQUENCE_LEN),

        # Warstwa Bi-LSTM: Czyta tekst w obu kierunkach (lewo->prawo i prawo->lewo)
        Bidirectional(LSTM(LSTM_UNITS, dropout=0.2, recurrent_dropout=0.2)),

        # Warstwa Wyjściowa: Jeden neuron, który wyrzuci nam prawdopodobieństwo od 0 do 1
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()  # Wyświetli ładną tabelkę z architekturą

    # 4. Trening Modelu
    print(f"\nRozpoczęto trening (Epoki: {EPOCHS}, Batch: {BATCH_SIZE})...")
    print("UWAGA: To potrwa ZNACZNIE dłużej niż SVM. Możesz iść zrobić kawę!")

    start_train_time = time.time()

    # fit() uruchamia proces uczenia. Parametr validation_split pozwala monitorować przeuczenie.
    history = model.fit(
        X_train_pad, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        verbose=1
    )

    end_train_time = time.time()
    train_time = end_train_time - start_train_time
    print(f"\n -> Czas treningu Bi-LSTM: {train_time:.4f} sekund")

    # 5. Inferencja (Testowanie)
    print("\nTrwa inferencja (przewidywanie na zbiorze testowym)...")

    start_infer_time = time.time()
    y_pred_probs = model.predict(X_test_pad)  # Zwraca prawdopodobieństwa np. 0.82
    end_infer_time = time.time()

    # Zamieniamy prawdopodobieństwo na twarde 0 lub 1 (próg = 0.5)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()

    infer_time = end_infer_time - start_infer_time
    infer_time_per_sample = (infer_time / len(y_test)) * 1000

    # ==============================================================================
    # NOWY BLOK: ANALIZA BŁĘDÓW DLA BI-LSTM (ERROR ANALYSIS)
    # ==============================================================================
    # Zbieramy wszystko do jednej tabeli, żeby łatwo to przeglądać w Excelu
    df_analysis = pd.DataFrame({
        'Wyczyszczona_Recenzja': X_test_text,
        'Faktyczna_Ocena': y_test,
        'Przewidywanie_BiLSTM': y_pred
    })

    # Oddzielamy sukcesy od porażek
    df_errors = df_analysis[df_analysis['Faktyczna_Ocena'] != df_analysis['Przewidywanie_BiLSTM']]
    df_success = df_analysis[df_analysis['Faktyczna_Ocena'] == df_analysis['Przewidywanie_BiLSTM']]

    # Definiujemy nazwy plików
    error_file = os.path.join(data_folder, f'bilstm_bledy_{language}.csv')
    success_file = os.path.join(data_folder, f'bilstm_sukcesy_{language}.csv')

    # Zapisujemy losową próbkę 50 przykładów (lub mniej, jeśli błędów jest mało)
    df_errors.sample(min(50, len(df_errors)), random_state=42).to_csv(error_file, index=False)
    df_success.sample(min(50, len(df_success)), random_state=42).to_csv(success_file, index=False)

    print(f"\nZapisano przykłady do analizy ręcznej w folderze: {data_folder}")
    print(f" -> Plik z błędami: bilstm_bledy_{language}.csv")
    print(f" -> Plik z sukcesami: bilstm_sukcesy_{language}.csv")
    # ==============================================================================


    print(f" -> Całkowity czas inferencji: {infer_time:.4f} sekund")
    print(f" -> Średni czas na 1 recenzję: {infer_time_per_sample:.4f} ms")

    # 6. Wyniki
    print("\n--- RAPORT WYNIKÓW BI-LSTM ---")
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')

    print(f"Accuracy (Dokładność): {acc:.4f}")
    print(f"F1-Score (Macro):      {f1:.4f}")
    #(Precision, Recall, F1 dla każdej klasy z osobna)
    print("\nSzczegółowy raport klasyfikacji:")
    print(classification_report(y_test, y_pred, target_names=['Negatywne (0)', 'Pozytywne (1)']))
    return {
        'Język': language.upper(),
        'Accuracy': acc,
        'F1-Score': f1,
        'Czas Treningu [s]': train_time,
        'Czas Inferencji [s]': infer_time
    }





