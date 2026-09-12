import os

from preprocessing import run_quota_preprocessing, run_max_balanced_preprocessing, run_parallel_preprocessing
from model_svm import train_and_evaluate_svm, pd
from model_bilstm import train_and_evaluate_bilstm
from raport_generating import generuj_raport, wygeneruj_porownanie

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
            '1091500', # Cyberpunk 2077
            '546560',  # Half-Life: Alyx
            '1145360', # Hades
            '782330',  # Doom Eternal
            '2208920', # Assassin's Creed Valhalla
            '1551360', # Forza Horizon 5
            '1086940', # Baldur's Gate 3
            '990080',  # Hogwarts Legacy
            '1774580', # Star Wars Jedi Survivor---- 1 odnotowany test
            '534380',  # Dying Light 2
            '1601580', # FrostPunk 2
            '1363080', # Manor Lords
            '1245620', # Elders Ring
            '1966720', # Lethal Company --- 2 odnotowany
            '1517290', # Battlefield 2042
            '1506830', # FIFA 22
            '1248130', # Farming Simulator 22
            '1190970', # House Flipper 2
            '1139900', # Ghostrunner
            '892970',  # Valheim
            '1326470', # Sons Of The Forest
            '261550',  # Mount & Blade II: Bannerlord -- 3 test
            '12210',   # Grand Theft Auto 4
            '239140',  # Dying Light
            '304390',  # For Honor
            '393380',  # Borderlands 3
            '553850',  # Hell Divers 2
            '1971870', # Mortal Kombat 1
            '2357570', # Overwatch
            '1599340', # Lost ArK
            '1468810', #Tale of immortal -4 test próba dostania 1500 recenzji na język oraz wynik (pozytywny/negatywny)
        ]
        # run_quota_preprocessing(
        #     app_ids_list=TARGET_APP_IDS,
        #     input_folder=INPUT_DIR,
        #     output_folder=OUTPUT_DIR
        # )
        #run_max_balanced_preprocessing(INPUT_DIR,OUTPUT_DIR, neg_pl_limit=20000)
        run_parallel_preprocessing(INPUT_DIR,OUTPUT_DIR, neg_pl_limit=20000)
    if stages["model_svm"]:
        PROCESSED_DATA_DIR = './processed_data/'
        results = []


        res_en = train_and_evaluate_svm(language='en', data_folder=PROCESSED_DATA_DIR)
        if res_en:
            results.append(res_en)


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
            df_results.to_csv(os.path.join(PROCESSED_DATA_DIR, 'wyniki_svm.csv'), index=False)

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
            df_results.to_csv(os.path.join(PROCESSED_DATA_DIR, 'wyniki_bilstm.csv'), index=False)
    if stages["raport"]:
        generuj_raport()
        wygeneruj_porownanie('pl')
        wygeneruj_porownanie('en')