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
    path_str = "../datasets/preprocessing/adult/adult-one-hot.csv"
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
    path_str = "../datasets/preprocessing/adult/adult-all-cat.csv"
    filepath = Path(path_str)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    adultData_all_cat.to_csv(filepath, index = False)

    """
        Generate a version of the dataset with a hybrid transformation with categorized with Label Encoder and one-hot-encoding (except target which we already transformed)
    """

    adultDataAux = pd.get_dummies(adultDataAux, prefix_sep="_", columns=not_trivial_cat_columns)
    adultDataAux = pd.concat([adultDataAux, classification_target], axis=1)
    path_str = "../datasets/preprocessing/adult/adult-hybrid.csv"
    filepath = Path(path_str)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    adultDataAux.to_csv(filepath, index = False)


    """
        Generate a versions of the all cat, one-hot, and hybrid datasets with each protected attribute as classification target
    """

    ## Age categories
    age_values = adultDataInfo.data["Age"].unique()
    age_cat_aux = ["Age_" + val for val in age_values]

    ## Race categories
    race_values = adultDataInfo.data["Race"].unique()
    race_cat_aux = ["Race_" + val for val in race_values]

    ## Sex categories
    sex_values = adultDataInfo.data["Sex"].unique()
    sex_cat_aux = ["Sex_" + val for val in sex_values]

    di_all_cat = DataInfo("../datasets/preprocessing/adult/adult-all-cat.csv", ["Age", "Race", "Sex"])
    di_all_cat.generate_all_protected_classification_csv()

    di_one_hot = DataInfo("../datasets/preprocessing/adult/adult-one-hot.csv", age_cat_aux + race_cat_aux + sex_cat_aux)
    di_one_hot.generate_all_protected_classification_csv()

    di_hybrid = DataInfo("../datasets/preprocessing/adult/adult-hybrid.csv", ["Age"] + race_cat_aux + sex_cat_aux)
    di_hybrid.generate_all_protected_classification_csv()
    

#==============================================================================

def ProcessCompas():
    print("==================== Preprocessing Compas Dataset ==================")
    print()

    compasDataInfo = DataInfo("../datasets/compas/compas.csv", ["African_American", "Asian", "Hispanic", "Native_American", "Other", "Female"])
    compasDataInfo.print_stats()
    compasDataInfo.generate_all_protected_classification_csv()
    print()

    # This dataset is already categorized, so nothing else to do
    


#==============================================================================

def ProcessGerman():
    print("==================== Preprocessing German Dataset ==================")
    print()

    germanDataInfo = DataInfo("../datasets/german-credit/german.csv", ["age", "age_cat", "foreign_worker"])
    germanDataInfo.print_stats()
    germanDataInfo.generate_all_protected_classification_csv()
    print()
    germanDataAux = germanDataInfo.data.copy()

    # Target is categorical (binary) with an order
    # values in:    "good"
    #               "bad"
    replace_class_map = {'class' : {'bad': 0, 'good': 1}}
    germanDataAux.replace(replace_class_map, inplace=True)

    # age_cat is categorical (binary) with an order
    # values in:    "young"
    #               "old"
    replace_age_cat_map = {'age_cat' : {'young' : 0,'old': 1}}
    germanDataAux.replace(replace_age_cat_map, inplace=True)

    # own_telephone is categorical (binary) with an order
    # values in:    "none"
    #               "yes"
    replace_own_telephone_map = {'own_telephone' : {'none' : 0,'yes': 1}}
    germanDataAux.replace(replace_own_telephone_map, inplace=True)

    # foreign_worker is categorical (binary) with an order
    # values in:    "no"
    #               "yes"
    replace_foreign_worker_map = {'foreign_worker' : {'no' : 0,'yes': 1}}
    germanDataAux.replace(replace_foreign_worker_map, inplace=True)


    classification_target = germanDataAux["class"]
    germanDataAux = germanDataAux.drop(["class"], axis=1)

    """
        Generate a version of the dataset with all the labels that are not integers binarized with one-hot-encoding (except class which we already transformed)
    """

    germanData_onehot = germanDataAux.copy()
    germanData_onehot = pd.get_dummies(germanData_onehot, prefix_sep="_")
    germanData_onehot = pd.concat([germanData_onehot, classification_target], axis=1)
    path_str = "../datasets/preprocessing/german/german-one-hot.csv"
    filepath = Path(path_str)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    germanData_onehot.to_csv(filepath, index = False)

    """
        Generate a version of the dataset with all the labels categorized with Label Encoder (except target which we already transformed)
    """

    # checking_status is categorical with an order
    # values in:    "no checking"
    #               "<0"
    #               "0<=X<200"
    #               ">=200"
    replace_checking_status_map = {'checking_status' : {'no checking': 0, '<0': 1, '0<=X<200' : 2, '>=200' : 3}}
    germanDataAux.replace(replace_checking_status_map, inplace=True)

    # savings_status is categorical with an order
    # values in:    "no known savings"
    #               "<100"
    #               "100<=X<500"
    #               "500<=X<1000"
    #               ">=1000"
    replace_savings_status_map = {'savings_status' : {'no known savings': 0, '<100': 1, '100<=X<500' : 2, '500<=X<1000': 3, '>=1000': 4}}
    germanDataAux.replace(replace_savings_status_map, inplace=True)

    # employment is categorical with an order
    # values in:    "unemployed"
    #               "<1"
    #               "1<=X<4"
    #               "4<=X<7"
    #               ">=7"
    replace_employment_map = {'employment' : {'unemployed': 0, '<1': 1, '1<=X<4' : 2, '4<=X<7' : 3, '>=7' : 4}}
    germanDataAux.replace(replace_employment_map, inplace=True)


    ## Label Encoder conversion
    germanData_all_cat = germanDataAux.copy()
    le = LabelEncoder()
    not_trivial_cat_columns = ["credit_history", "purpose", "personal_status", "other_parties", "property_magnitude", "other_payment_plans", "housing", "job"]
    for label in not_trivial_cat_columns:
        germanData_all_cat[label] = le.fit_transform(germanData_all_cat[label])
    germanData_all_cat = pd.concat([germanData_all_cat, classification_target], axis=1)
    path_str = "../datasets/preprocessing/german/german-all-cat.csv"
    filepath = Path(path_str)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    germanData_all_cat.to_csv(filepath, index = False)

    """
        Generate a version of the dataset with a hybrid transformation with categorized with Label Encoder and one-hot-encoding (except target which we already transformed)
    """

    germanDataAux = pd.get_dummies(germanDataAux, prefix_sep="_", columns=not_trivial_cat_columns)
    germanDataAux = pd.concat([germanDataAux, classification_target], axis=1)
    path_str = "../datasets/preprocessing/german/german-hybrid.csv"
    filepath = Path(path_str)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    germanDataAux.to_csv(filepath, index = False)


    """
        Generate a versions of the all cat, one-hot, and hybrid datasets with each protected attribute as classification target
    """

    di_all_cat = DataInfo("../datasets/preprocessing/german/german-one-hot.csv", ["age", "age_cat", "foreign_worker"])
    di_all_cat.generate_all_protected_classification_csv()

    di_one_hot = DataInfo("../datasets/preprocessing/german/german-all-cat.csv", ["age", "age_cat", "foreign_worker"])
    di_one_hot.generate_all_protected_classification_csv()

    di_hybrid = DataInfo("../datasets/preprocessing/german/german-hybrid.csv", ["age", "age_cat", "foreign_worker"])
    di_hybrid.generate_all_protected_classification_csv()

#==============================================================================

def ProcessTitanic():
    print("==================== Preprocessing Titanic Dataset ==================")
    print()

    # This dataset has the classification target "Survived" as the first column
    # This dataset has a column "Name" that is unique, id like
    # No other transformations necessary

    titanicDataInfo = DataInfo("../datasets/titanic/titanic.csv", ["Sex","Age"])
    classification_target = titanicDataInfo.data["Survived"]
    titanicDataInfo.data = titanicDataInfo.data.drop(["Survived"], axis=1)
    titanicDataInfo.data = titanicDataInfo.data.drop(["Name"], axis=1)

    # Sex is categorical (binary)
    # values in:    "male"
    #               "female"
    replace_sex_map = {'Sex' : {'male' : 1,'female': 0}}
    titanicDataInfo.data.replace(replace_sex_map, inplace=True)

    titanicDataInfo.data = pd.concat([titanicDataInfo.data, classification_target], axis=1)
    path_str = "../datasets/preprocessing/titanic/titanic-no-name.csv"
    filepath = Path(path_str)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    titanicDataInfo.data.to_csv(filepath, index = False)

    di_titanic = DataInfo("../datasets/preprocessing/titanic/titanic-no-name.csv", ["Sex", "Age"])
    di_titanic.print_stats()
    di_titanic.generate_all_protected_classification_csv()
    print()

#==============================================================================

def ProcessRicci():
    print("==================== Preprocessing Ricci Dataset ==================")
    print()

    # see https://rdrr.io/cran/Stat2Data/man/Ricci.html

    ricciDataInfo = DataInfo("../datasets/ricci/ricci.csv", ["Race"])
    ricciDataInfo.print_stats()
    ricciDataInfo.generate_all_protected_classification_csv()
    print()
    ricciDataAux = ricciDataInfo.data.copy()

    classification_target = ricciDataAux["Class"]
    ricciDataAux = ricciDataAux.drop(["Class"], axis=1)

    """
        Generate a version of the dataset with all the labels binarized with one-hot-encoding
    """
    ricciData_onehot = ricciDataAux.copy()
    ricciData_onehot = pd.get_dummies(ricciData_onehot, prefix_sep="_")
    ricciData_onehot = pd.concat([ricciData_onehot, classification_target], axis=1)
    path_str = "../datasets/preprocessing/ricci/ricci-one-hot.csv"
    filepath = Path(path_str)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    ricciData_onehot.to_csv(filepath, index = False)

    """
        Generate a version of the dataset with all the labels categorized with Label Encoder (except target which we already transformed)
    """

    ## Label Encoder conversion
    le = LabelEncoder()
    not_trivial_cat_columns = ["Position", "Race"]
    for label in not_trivial_cat_columns:
        ricciDataAux[label] = le.fit_transform(ricciDataAux[label])
    ricciDataAux = pd.concat([ricciDataAux, classification_target], axis=1)
    path_str = "../datasets/preprocessing/ricci/ricci-all-cat.csv"
    filepath = Path(path_str)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    ricciDataAux.to_csv(filepath, index = False)

    # there is no point on an hybrid generation


    """
        Generate a versions of the all cat, and one-hot datasets with each protected attribute as classification target
    """

    ## Race categories
    race_values = ricciDataInfo.data["Race"].unique()
    race_cat_aux = ["Race_" + val for val in race_values]


    di_all_cat = DataInfo("../datasets/preprocessing/ricci/ricci-all-cat.csv", ["Race"])
    di_all_cat.generate_all_protected_classification_csv()

    di_one_hot = DataInfo("../datasets/preprocessing/ricci/ricci-one-hot.csv", race_cat_aux)
    di_one_hot.generate_all_protected_classification_csv()



#==============================================================================

def ProcessLawsuit():
    print("==================== Preprocessing Lawsuit Dataset ==================")
    print()

    # see https://www.kaggle.com/datasets/hjmjerry/gender-discrimination
    # there is no classification column



#==============================================================================

if __name__ == '__main__':
    ProcessAdult()
    ProcessCompas()
    ProcessGerman()
    ProcessTitanic()
    ProcessRicci()
