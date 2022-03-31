import pandas as pd
#from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
from DataInfo import *


#==============================================================================

def ProcessAdult():
    print("==================== Preprocessing Adult Dataset ==================")
    print()

    adultDataInfo = DataInfo("../datasets/adult/adult.csv", ["Age","Race","Sex"])
    adultDataInfo.print_stats()
    adultDataInfo.generate_all_protected_classification_csv()
    print()
    adultDataAux = adultDataInfo.data.copy()

    # Target is categorical (binary) with an order
    # values in:    "Less than $50,000"
    #               "More than $50,000"
    replace_target_map = {'Target' : {'Less than $50,000': 0, 'More than $50,000': 1}}
    adultDataAux.replace(replace_target_map, inplace=True)

    classification_target = adultDataAux["Target"]
    adultDataAux = adultDataAux.drop(["Target"], axis=1)

    """
        Generate a version of the dataset with all the labels binarized with one-hot-encoding (except target which we already transformed)
    """
    adultData_onehot = adultDataAux.copy()
    adultData_onehot = pd.get_dummies(adultData_onehot, prefix_sep="_")
    adultData_onehot = pd.concat([adultData_onehot, classification_target], axis=1)
    path_str = "results/adult/adult-one-hot.csv"
    filepath = Path(path_str)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    adultData_onehot.to_csv(filepath, index = False)

    """
        Generate a version of the dataset with all the labels categorized with Label Encoder (except target which we already transformed)
    """

    # Age is categorical with an order
    # values in:    "Age <= 28.00"
    #               "28.00 < Age <= 37.00"
    #               "37.00 < Age <= 48.00"
    #               "Age > 48.00"
    replace_age_map = {'Age' : {'Age <= 28.00': 1, '28.00 < Age <= 37.00': 2, '37.00 < Age <= 48.00' : 3, 'Age > 48.00' : 4}}
    adultDataAux.replace(replace_age_map, inplace=True)

    # Capital Gain is categorical with an order
    # values in:    "None"
    #               "Low"
    #               "High"
    replace_capitalgain_map = {'Capital Gain' : {'None': 0, 'Low': 1, 'High' : 2}}
    adultDataAux.replace(replace_capitalgain_map, inplace=True)

    # Capital Loss is categorical with an order
    # values in:    "None"
    #               "Low"
    #               "High"
    replace_capitalloss_map = {'Capital Loss' : {'None': 0, 'Low': 1, 'High' : 2}}
    adultDataAux.replace(replace_capitalloss_map, inplace=True)

    # Hours per week is categorical with an order
    # values in:    "Hours per week <= 40.00"
    #               "40.00 < Hours per week <= 45.00"
    #               "Hours per week > 45.00"
    #               "Age > 48.00"
    replace_hpw_map = {'Hours per week' : {'Hours per week <= 40.00': 1, '40.00 < Hours per week <= 45.00': 2, 'Hours per week > 45.00' : 3}}
    adultDataAux.replace(replace_hpw_map, inplace=True)

    ## Label Encoder conversion
    adultData_all_cat = adultDataAux.copy()
    le = LabelEncoder()
    not_trivial_cat_columns = ["Workclass", "Education", "Marital Status", "Occupation", "Relationship", "Race", "Sex", "Country"]
    for label in not_trivial_cat_columns:
        adultData_all_cat[label] = le.fit_transform(adultData_all_cat[label])
    adultData_all_cat = pd.concat([adultData_all_cat, classification_target], axis=1)
    path_str = "results/adult/adult-all-cat.csv"
    filepath = Path(path_str)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    adultData_all_cat.to_csv(filepath, index = False)

    """
        Generate a version of the dataset with a hybrid transformation with categorized with Label Encoder and one-hot-encoding (except target which we already transformed)
    """

    adultDataAux = pd.get_dummies(adultDataAux, prefix_sep="_", columns=not_trivial_cat_columns)
    adultDataAux = pd.concat([adultDataAux, classification_target], axis=1)
    path_str = "results/adult/adult-hybrid.csv"
    filepath = Path(path_str)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    adultDataAux.to_csv(filepath, index = False)


    """
        Generate a versions of the all cat, one-hot and hybrid datasets with each protected attribute as classification target
    """

    ## Age categories
    age_values = adultDataInfo.data["Age"].unique()
    age_cat_aux = ["Age_" + val for val in age_values]

    ## Race categories
    race_values = adultDataInfo.data["Race"].unique()
    race_cat_aux = ["Race_" + val for val in race_values]

    ## Race categories
    sex_values = adultDataInfo.data["Sex"].unique()
    sex_cat_aux = ["Sex_" + val for val in sex_values]

    di_all_cat = DataInfo("results/adult/adult-all-cat.csv", ["Age", "Race", "Sex"])
    di_all_cat.generate_all_protected_classification_csv()

    di_one_hot = DataInfo("results/adult/adult-one-hot.csv", age_cat_aux + race_cat_aux + sex_cat_aux)
    di_one_hot.generate_all_protected_classification_csv()

    di_hybrid = DataInfo("results/adult/adult-hybrid.csv", ["Age"] + race_cat_aux + sex_cat_aux)
    di_hybrid.generate_all_protected_classification_csv()
    



#==============================================================================

if __name__ == '__main__':
    ProcessAdult()