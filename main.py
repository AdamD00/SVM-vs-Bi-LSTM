from preprocessing import run_quota_preprocessing
from model_svm import train_and_evaluate_svm, pd
from model_bilstm import train_and_evaluate_bilstm
from raport_generating import generuj_raport

if __name__ == "__main__":

    stages = {
        "preprocessing" : True,
        "model_svm" : True,
        "model_bilstm": True,
        "raport" : True
    }
    print(stages)
    if stages["preprocessing"]:
        INPUT_DIR = './dane/Game Reviews/'
        OUTPUT_DIR = './processed_data/'

        TARGET_APP_IDS = [
            '1091500',  # Cyberpunk 2077
            '546560',  # Half-Life: Alyx
            '1145360', # Hades
            '782330', # Doom Eternal
            '2208920', # Assasins Creed Valhalla
            '1551360', # Forza Horizon 5
            '1086940', # Baldurs Gate 3
            '990080', # Hogwart Legacy
            '1774580', # Star Wars Jedi Survivor
            '534380', # Dying Light 2
            '1601580', # FrostPunk 2
            '1363080', # Manor Lords
            '1245620', # Elders Ring
            '1966720', # Lethal Company
        ]
        run_quota_preprocessing(
            app_ids_list=TARGET_APP_IDS,
            input_folder=INPUT_DIR,
            output_folder=OUTPUT_DIR,
            target_per_language=2900
        )
    if stages["model_svm"]:
        PROCESSED_DATA_DIR = './processed_data/'
        results = []

        # Najpierw Język Angielski
        res_en = train_and_evaluate_svm(language='en', data_folder=PROCESSED_DATA_DIR)
        if res_en:
            results.append(res_en)

        # Potem Język Polski
        res_pl = train_and_evaluate_svm(language='pl', data_folder=PROCESSED_DATA_DIR)
        if res_pl:
            results.append(res_pl)

        # Podsumowanie w formie tabeli
        if results:
            print("\n\n" + "=" * 50)
            print("ZBIORCZE PODSUMOWANIE EKSPERYMENTU SVM")
            print("=" * 50)
            df_results = pd.DataFrame(results)
            print(df_results.to_markdown(index=False, floatfmt=".4f"))

    if stages["model_bilstm"]:
        PROCESSED_DATA_DIR = './processed_data/'
        results = []

        res_en = train_and_evaluate_bilstm(language='en', data_folder=PROCESSED_DATA_DIR)
        if res_en: results.append(res_en)

        res_pl = train_and_evaluate_bilstm(language='pl', data_folder=PROCESSED_DATA_DIR)
        if res_pl: results.append(res_pl)

        if results:
            print("\n\n" + "=" * 50)
            print("ZBIORCZE PODSUMOWANIE EKSPERYMENTU BI-LSTM")
            print("=" * 50)
            df_results = pd.DataFrame(results)
            print(df_results.to_markdown(index=False, floatfmt=".4f"))
    if stages["raport"]:
        generuj_raport()