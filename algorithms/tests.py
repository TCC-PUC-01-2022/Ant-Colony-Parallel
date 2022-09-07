import pandas as pd
import numpy as np
import hashlib
from ant_is import run_colony
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE 
from collections import Counter

import json
from typing import *

def get_stratfied_cross_validation_scores(scores: List[Dict]):
    new_score = scores[0].copy()
    keys = new_score.keys()
    test_length = len(scores)
    for key in keys:
        if isinstance(new_score[key], dict):
            for sub_key in new_score[key].keys():
                value = sum([i[key][sub_key] for i in scores]) / test_length
                new_score[key][sub_key] = value
        else:
            value = sum([i[key] for i in scores]) / test_length
            new_score[key] = value

    return new_score

def create_test(classifier,
                num_iterations: int,
                num_folds: int,
                X: np.ndarray,
                y: np.ndarray,
                X_valid,
                y_valid,
                output_name: str,
                reduce_instances: bool = False,
                initial_pheromone: float = None,
                evaporation_rate: float = None,
                Q: float = None,
                resample_data = False ):
    
    results = {}
    for i in range(num_iterations):
        print(f"beginning of iterarion {i}")
        folds = StratifiedKFold(n_splits=num_folds, shuffle=True)
        partial_scores = []
        valid_scores = []
        selected_indices = []
        red_ratios = []
        y_ratios = []
        class_infos  = []
        total_instances_ori = []
        total_instances_red = []
        total_instances_res = []
        for train_indices, test_indices in folds.split(X, y):
            class_info = {}
            num_instances_ori = len(train_indices)
            num_instances_reduced = -1
            num_instances_resampled = -1

            train_data = X[train_indices]
            train_labels = y[train_indices]
            
            counter = Counter(train_labels)
            class_info["num_superior_original"] = counter['Superior']
            class_info["num_inferior_original"] = counter['Inferior']
            class_info["sup_rate_original"] = counter['Superior'] / train_labels.size
            class_info["inf_rate_original"] = counter['Inferior'] / train_labels.size
            
            if reduce_instances:
                reduced_indices = run_colony(train_data, train_labels, initial_pheromone, evaporation_rate, Q)
                hashed_instances = []
                for indice in reduced_indices:
                    x_indice = X[indice, :].tolist()
                    y_indice = y[indice]
                    x_indice.append(y_indice)
                    hashed_instance = hashlib.sha256(str(x_indice).encode('utf-8')).hexdigest()
                    hashed_instances.append(hashed_instance)

                selected_indices.append(hashed_instances)
                red_ratios.append(reduced_indices.size / train_indices.size)

                train_data = X[reduced_indices]
                train_labels = y[reduced_indices]

                num_instances_reduced = reduced_indices.size
                
                counter = Counter(train_labels)
                class_info["num_superior_reduced"] = counter['Superior']
                class_info["num_inferior_reduced"] = counter['Inferior']
                class_info["sup_rate_reduced"] = counter['Superior'] / train_labels.size
                class_info["inf_rate_reduced"] = counter['Inferior'] / train_labels.size

            if resample_data:
                smote = SMOTE()
                train_data, train_labels = smote.fit_resample(train_data, train_labels)
                
                num_instances_resampled = train_labels.size
                
                counter = Counter(train_labels)
                class_info["num_superior_resampled"] = counter['Superior']
                class_info["num_inferior_resampled"] = counter['Inferior']
                class_info["sup_rate_resampled"] = counter['Superior'] / train_labels.size
                class_info["inf_rate_resampled"] = counter['Inferior'] / train_labels.size

            classifier.fit(train_data, train_labels)
            y_pred = classifier.predict(X[test_indices])
            y_pred_valid = classifier.predict(X_valid)

            score = classification_report(y[test_indices], y_pred, output_dict=True, zero_division=1)
            valid_score = classification_report(y_valid, y_pred_valid, output_dict=True, zero_division=1)

            valid_scores.append(valid_score)
            partial_scores.append(score)
            class_infos.append(class_info)
            total_instances_ori.append(num_instances_ori)
            total_instances_red.append(num_instances_reduced)
            total_instances_res.append(num_instances_resampled)

        scores = get_stratfied_cross_validation_scores(partial_scores)
        v_score = get_stratfied_cross_validation_scores(valid_scores)
        print(f"accuracy: Valid: {scores['accuracy']} -- Test: {v_score['accuracy']}")
        for metric_name in scores["macro avg"].keys():
            print(f"{metric_name}: Valid: {scores['macro avg'][metric_name]} -- Test: {v_score['macro avg'][metric_name]}")

        print("-------------------------------------------------------")
        new_result = "i" + str(i)
        results[new_result] = {
            "scores": scores,
            "valid_scores": v_score,
            "partial_scores": partial_scores,
            "partial_valid_scores": valid_scores,
            "class_ratios": y_ratios,
            "class_info": class_infos,
            "total_instances_ori": total_instances_ori,
            "total_instances_red": total_instances_red,
            "total_instances_res": total_instances_res
        }

        if reduce_instances:
            results[new_result]["selected_indices"] = selected_indices
            results[new_result]["reduction_ratios"] = red_ratios

    results["num_iterations"] = num_iterations
    results["num_folds"] = num_folds
    results["strategy"] = "Strafied Cross Validation" 

    print("Generating output JSON file....")
    json.dump(results, open(output_name, 'w'), indent=2)

def run_all_tests(X, y, X_valid, y_valid, term_output):
    print("RUNNING TEST FOR " + term_output)

    print("1-NN Test FULL")
    classifier = KNeighborsClassifier(n_neighbors=1)
    create_test(classifier, 1, 10, X, y, X_valid, y_valid, f'outputs/full_{term_output}_1nn_results.json', False, 1, 0.1, 1)

    print("1-NN Test Resampled")
    classifier = KNeighborsClassifier(n_neighbors=1)
    create_test(classifier, 1, 10, X, y, X_valid, y_valid, f'outputs/resampled_{term_output}_1nn_results.json', False, 1, 0.1, 1, True)
    
    print("1-NN Test Reduced")
    classifier = KNeighborsClassifier(n_neighbors=1)
    create_test(classifier, 10, 10, X, y, X_valid, y_valid, f'outputs/reduced_{term_output}_1nn_results.json', True, 1, 0.1, 1)

    print("1-NN Test reduced Resampled")
    classifier = KNeighborsClassifier(n_neighbors=1)
    create_test(classifier, 10, 10, X, y, X_valid, y_valid, f'outputs/improved_{term_output}_1nn_results.json', True, 1, 0.1, 1, True)

    print("Gaussian NB Full")
    classifier = GaussianNB()
    create_test(classifier, 1, 10, X, y, X_valid, y_valid, f'outputs/full_{term_output}_nb_results.json', False, 1, 0.1, 1)
    
    print("Gaussian NB Resampled")
    classifier = GaussianNB()
    create_test(classifier, 1, 10, X, y, X_valid, y_valid, f'outputs/resampled_{term_output}_nb_results.json', False, 1, 0.1, 1, True)

    print("Gaussian NB Reduced")
    classifier = GaussianNB()
    create_test(classifier, 10, 10, X, y, X_valid, y_valid, f'outputs/reduced_{term_output}_nb_results.json', True, 1, 0.1, 1)
    
    print("Gaussian NB Reduced Resampled")
    classifier = GaussianNB()
    create_test(classifier, 10, 10, X, y, X_valid, y_valid, f'outputs/improved_{term_output}_nb_results.json', True, 1, 0.1, 1, True)

    print("CART test Full")
    classifier = DecisionTreeClassifier()
    create_test(classifier, 1, 10, X, y, X_valid, y_valid, f'outputs/full_{term_output}_cart_results.json', False, 1, 0.1, 1)
    
    print("CART test Resampled")
    classifier = DecisionTreeClassifier()
    create_test(classifier, 1, 10, X, y, X_valid, y_valid, f'outputs/resampled_{term_output}_cart_results.json', False, 1, 0.1, 1, True)

    print("CART test Reduced Resampled")
    classifier = DecisionTreeClassifier()
    create_test(classifier, 10, 10, X, y, X_valid, y_valid, f'outputs/reduced_{term_output}_cart_results.json', True, 1, 0.1, 1)
    
    print("CART test Reduced")
    classifier = DecisionTreeClassifier()
    create_test(classifier, 10, 10, X, y, X_valid, y_valid, f'outputs/improved_{term_output}_cart_results.json', True, 1, 0.1, 1, True)

    print("SVM Test Full")
    classifier = SVC()
    create_test(classifier, 1, 10, X, y, X_valid, y_valid, f'outputs/full_{term_output}_svm_results.json', False, 1, 0.1, 1)
    
    print("SVM Test Resampled")
    classifier = SVC()
    create_test(classifier, 1, 10, X, y, X_valid, y_valid, f'outputs/resampled_{term_output}_svm_results.json', False, 1, 0.1, 1, True)

    print("SVM Test Reduced")
    classifier = SVC()
    create_test(classifier, 10, 10, X, y, X_valid, y_valid, f'outputs/reduced_{term_output}_svm_results.json', True, 1,0.1, 1)
    
    print("SVM Test Reduced Resampled")
    classifier = SVC()
    create_test(classifier, 10, 10, X, y, X_valid, y_valid, f'outputs/improved_{term_output}_svm_results.json', True, 1,0.1, 1, True)

    print("MLP Test Full")
    classifier = MLPClassifier()
    create_test(classifier, 1, 10, X, y, X_valid, y_valid, f'outputs/full_{term_output}_mlp_results.json', False, 1,0.1, 1)
    
    print("MLP Test Resampled")
    classifier = MLPClassifier()
    create_test(classifier, 1, 10, X, y, X_valid, y_valid, f'outputs/resampled_{term_output}_mlp_results.json', False, 1,0.1, 1, True)

    print("MLP Test Reduced")
    classifier = MLPClassifier()
    create_test(classifier, 10, 10, X, y, X_valid, y_valid, f'outputs/reduced_{term_output}_mlp_results.json', True, 1,0.1, 1)
    
    print("MLP Test Reduced Resampled")
    classifier = MLPClassifier()
    create_test(classifier, 10, 10, X, y, X_valid, y_valid, f'outputs/improved_{term_output}_mlp_results.json', True, 1,0.1, 1, True)

    print("Random Forest Test Full")
    classifier = RandomForestClassifier()
    create_test(classifier, 1, 10, X, y, X_valid, y_valid, f'outputs/full_{term_output}_random_forest_results.json', False, 1,0.1, 1)
    
    print("Random Forest Test Resampled")
    classifier = RandomForestClassifier()
    create_test(classifier, 1, 10, X, y, X_valid, y_valid, f'outputs/resampled_{term_output}_random_forest_results.json', False, 1,0.1, 1, True)

    print("Random Forest Test Reduced")
    classifier = RandomForestClassifier()
    create_test(classifier, 10, 10, X, y, X_valid, y_valid, f'outputs/reduced_{term_output}_random_forest_results.json', True, 1,0.1, 1)
    
    print("Random Forest Test Reduced Resampled")
    classifier = RandomForestClassifier()
    create_test(classifier, 10, 10, X, y, X_valid, y_valid, f'outputs/improved_{term_output}_random_forest_results.json', True, 1,0.1, 1, True)



def test_smote(X, y):
    print(f"Shape of original data: {X.shape} {y.shape}")
    print(f"Count of original labels: {Counter(y)}")
    smote = SMOTE()
    X_res, y_res = smote.fit_resample(X, y)
    print(f"Shape of resampled data: {X_res.shape} {y_res.shape}")
    print(f"Count of resampled labels: {Counter(y_res)}")



def main():
    
    print('-----======PRE-PROCESSING-------=========')
    base_path = "databases/AG_Random_Especialista"
    df_arit = pd.read_csv(base_path + "/Aritmética/TreinamentoAritPrep.csv", sep=";")
    df_esc = pd.read_csv(base_path + "/Escrita/TreinamentoEscPrep.csv", sep=";")
    df_leit = pd.read_csv(base_path + "/Leitura/TreinamentoLeitPrep.csv", sep=";")

    df_valid_arit = pd.read_csv(base_path + "/Aritmética/TesteAritPrep.csv", sep=";")
    df_valid_esc = pd.read_csv(base_path + "/Escrita/TesteEscPrep.csv", sep=";")
    df_valid_leit = pd.read_csv(base_path + "/Leitura/TesteLeitPrep.csv", sep=";")

    # Test Arit
    df = df_arit.copy()
    df_valid = df_valid_arit.copy()
    y = df["TDE_MG_Arit"].to_numpy()
    X = df.drop(columns=["TDE_MG_Arit"]).to_numpy()
    y_valid = df_valid["TDE_MG_Arit"].to_numpy()
    X_valid = df_valid.drop(columns=["TDE_MG_Arit"]).to_numpy()
    run_all_tests(X, y, X_valid, y_valid, "arit")

    # Test Esc
    df = df_esc.copy()
    df_valid = df_valid_esc.copy()
    y = df["TDE_MG_Esc"].to_numpy()
    X = df.drop(columns=["TDE_MG_Esc"]).to_numpy()
    y_valid = df_valid["TDE_MG_Esc"].to_numpy()
    X_valid = df_valid.drop(columns=["TDE_MG_Esc"]).to_numpy()
    run_all_tests(X, y, X_valid, y_valid, "esc")

    # Test leit
    df = df_leit.copy()
    df_valid = df_valid_leit.copy()
    y = df["TDE_MG_Leit"].to_numpy()
    X = df.drop(columns=["TDE_MG_Leit"]).to_numpy()
    y_valid = df_valid["TDE_MG_Leit"].to_numpy()
    X_valid = df_valid.drop(columns=["TDE_MG_Leit"]).to_numpy()
    run_all_tests(X, y, X_valid, y_valid, "leit")


if __name__ == '__main__':
    main()
