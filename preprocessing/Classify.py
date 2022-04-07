import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_selector as selector
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer



def classifyLogisticRegression(data, target):

    data_lg = data.copy()
    numerical_columns_selector = selector(dtype_exclude=object)
    categorical_columns_selector = selector(dtype_include=object)
    numerical_columns = numerical_columns_selector(data_lg)
    categorical_columns = categorical_columns_selector(data_lg)
    numerical_preprocessor = StandardScaler()
    preprocessor = ColumnTransformer([
        ('standard_scaler', numerical_preprocessor, numerical_columns)])

    data_train, data_test, target_train, target_test = train_test_split(data_lg, target, random_state=42)
    model = make_pipeline(preprocessor, LogisticRegression(max_iter=500))
    _ = model.fit(data_train, target_train)

    score = model.score(data_test, target_test)

    print("\tLogistic Regression score: " + f"{score:.3f}")

    cv_results = cross_validate(model, data, target, cv=5)
    scores = cv_results["test_score"]
    print("\t\tThe mean cross-validation accuracy is: "
        f"{scores.mean():.3f} +/- {scores.std():.3f}")
    print()


def classifyLogisticRegressionMultinomial(data, target):

    data_lg = data.copy()
    numerical_columns_selector = selector(dtype_exclude=object)
    categorical_columns_selector = selector(dtype_include=object)
    numerical_columns = numerical_columns_selector(data_lg)
    categorical_columns = categorical_columns_selector(data_lg)
    numerical_preprocessor = StandardScaler()
    preprocessor = ColumnTransformer([
        ('standard_scaler', numerical_preprocessor, numerical_columns)])

    data_train, data_test, target_train, target_test = train_test_split(data_lg, target, random_state=42)
    model = make_pipeline(preprocessor, LogisticRegression(max_iter=500, multi_class="multinomial"))
    _ = model.fit(data_train, target_train)

    score = model.score(data_test, target_test)

    print("\tLogistic Regression Multinomial score: " + f"{score:.3f}")

    cv_results = cross_validate(model, data, target, cv=5)
    scores = cv_results["test_score"]
    print("\t\tThe mean cross-validation accuracy is: "
        f"{scores.mean():.3f} +/- {scores.std():.3f}")
    print()



def classifyHistGradientBoostingClassifier(data, target):
    data_train, data_test, target_train, target_test = train_test_split(data, target, random_state=42)
    model = HistGradientBoostingClassifier()
    _ = model.fit(data_train, target_train)

    score = model.score(data_test, target_test)

    print("\tHist Gradient Boost Classifier score: " + f"{score:.3f}")

    cv_results = cross_validate(model, data, target, cv=5)
    scores = cv_results["test_score"]
    print("\t\tThe mean cross-validation accuracy is: "
        f"{scores.mean():.3f} +/- {scores.std():.3f}")
    print()


def classifySVC(data, target):
    data_train, data_test, target_train, target_test = train_test_split(data, target, random_state=42)
    model = SVC()
    _ = model.fit(data_train, target_train)

    score = model.score(data_test, target_test)

    print("\tSupport Vector Classifier score: " + f"{score:.3f}")

    cv_results = cross_validate(model, data, target, cv=5)
    scores = cv_results["test_score"]
    print("\t\tThe mean cross-validation accuracy is: "
        f"{scores.mean():.3f} +/- {scores.std():.3f}")
    print()


def classifyDecisionTreeClassifier(data, target, to_print=False, to_draw=False, out_draw = "tree.dot"):
    data_train, data_test, target_train, target_test = train_test_split(data, target, random_state=42)
    model = DecisionTreeClassifier()
    _ = model.fit(data_train, target_train)

    score = model.score(data_test, target_test)

    print("\tDecision Tree Classifier score: " + f"{score:.3f}")

    cv_results = cross_validate(model, data, target, cv=5)
    scores = cv_results["test_score"]
    print("\t\tThe mean cross-validation accuracy is: "
        f"{scores.mean():.3f} +/- {scores.std():.3f}")
    print()

    """
        Print tree
    """
    if to_print:
        n_nodes = model.tree_.node_count
        children_left = model.tree_.children_left
        children_right = model.tree_.children_right
        feature = model.tree_.feature
        threshold = model.tree_.threshold

        node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
        is_leaves = np.zeros(shape=n_nodes, dtype=bool)
        stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
        while len(stack) > 0:
            # `pop` ensures each node is only visited once
            node_id, depth = stack.pop()
            node_depth[node_id] = depth

            # If the left and right child of a node is not the same we have a split
            # node
            is_split_node = children_left[node_id] != children_right[node_id]
            # If a split node, append left and right children and depth to `stack`
            # so we can loop through them
            if is_split_node:
                stack.append((children_left[node_id], depth + 1))
                stack.append((children_right[node_id], depth + 1))
            else:
                is_leaves[node_id] = True

        print(
            "\t\tThe binary tree structure has {n} nodes and has "
            "the following tree structure:\n".format(n=n_nodes)
        )
        for i in range(n_nodes):
            if is_leaves[i]:
                print(
                    "{space}node={node} is a leaf node.".format(
                        space=node_depth[i] * "\t", node=i
                    )
                )
            else:
                print(
                    "{space}node={node} is a split node: "
                    "go to node {left} if X[:, {feature}] <= {threshold} "
                    "else to node {right}.".format(
                        space=node_depth[i] * "\t",
                        node=i,
                        left=children_left[i],
                        feature=feature[i],
                        threshold=threshold[i],
                        right=children_right[i],
                    )
                )
        
        if to_draw:
            dot_data = tree.export_graphviz(model, out_file = out_draw)
            #$ dot -Tps tree.dot -o tree.ps      (PostScript format)
            #$ dot -Tpng tree.dot -o tree.png    (PNG format)



def classifyKNeighborsClassifier(data, target):
    data_train, data_test, target_train, target_test = train_test_split(data, target, random_state=42)
    model = KNeighborsClassifier()
    _ = model.fit(data_train, target_train)

    score = model.score(data_test, target_test)

    print("\tKNeighbors Classifier score: " + f"{score:.3f}")

    cv_results = cross_validate(model, data, target, cv=5)
    scores = cv_results["test_score"]
    print("\t\tThe mean cross-validation accuracy is: "
        f"{scores.mean():.3f} +/- {scores.std():.3f}")
    print()



def classifyNaiveBayesClassifier(data, target):
    data_train, data_test, target_train, target_test = train_test_split(data, target, random_state=42)
    model = GaussianNB()
    _ = model.fit(data_train, target_train)

    score = model.score(data_test, target_test)

    print("\tNayve Bayes Classifier score: " + f"{score:.3f}")

    cv_results = cross_validate(model, data, target, cv=5)
    scores = cv_results["test_score"]
    print("\t\tThe mean cross-validation accuracy is: "
        f"{scores.mean():.3f} +/- {scores.std():.3f}")
    print()


def classifyNNMLPClassifier(data, target):
    data_train, data_test, target_train, target_test = train_test_split(data, target, random_state=42)
    model = MLPClassifier(max_iter=500)
    _ = model.fit(data_train, target_train)

    score = model.score(data_test, target_test)

    print("\tNeural Network Multi-Layer Perceptron Classifier score: " + f"{score:.3f}")

    cv_results = cross_validate(model, data, target, cv=5)
    scores = cv_results["test_score"]
    print("\t\tThe mean cross-validation accuracy is: "
        f"{scores.mean():.3f} +/- {scores.std():.3f}")
    print()


#==============================================================================

def extractDataTarget(datafile):
    data = pd.read_csv(datafile)
    target_label = data.columns[-1]
    y = data[target_label]
    x = data.drop([target_label], axis=1)
    return x,y

#==============================================================================

def processAdult():

    print("==================== Preprocessing Adult Dataset ====================")
    print()

    print("### Age as proxy target")
    print()
    files = ["../datasets/preprocessing/adult-all-cat/adult-all-cat-Age.csv",
        "../datasets/preprocessing/adult-hybrid/adult-hybrid-Age.csv",
        "../datasets/preprocessing/adult-one-hot/adult-one-hot-Age_Age <= 28.00.csv",
        "../datasets/preprocessing/adult-one-hot/adult-one-hot-Age_28.00 < Age <= 37.00.csv",
        "../datasets/preprocessing/adult-one-hot/adult-one-hot-Age_37.00 < Age <= 48.00.csv",
        "../datasets/preprocessing/adult-one-hot/adult-one-hot-Age_Age > 48.00.csv"]
    
    for f in files:
        print("## file: " + f)
        data, target = extractDataTarget(f)
        classifyLogisticRegression(data, target)
        classifyLogisticRegressionMultinomial(data, target)
        classifyHistGradientBoostingClassifier(data, target)
        classifySVC(data, target)
        classifyDecisionTreeClassifier(data, target)
        classifyKNeighborsClassifier(data, target)
        classifyNaiveBayesClassifier(data, target)
        classifyNNMLPClassifier(data, target)

    print()
    print("### Race as proxy target")
    print()
    files = ["../datasets/preprocessing/adult-all-cat/adult-all-cat-Race.csv",
        "../datasets/preprocessing/adult-hybrid/adult-hybrid-Race_Amer-Indian-Eskimo.csv",
        "../datasets/preprocessing/adult-hybrid/adult-hybrid-Race_Asian-Pac-Islander.csv",
        "../datasets/preprocessing/adult-hybrid/adult-hybrid-Race_Black.csv",
        "../datasets/preprocessing/adult-hybrid/adult-hybrid-Race_Other.csv",
        "../datasets/preprocessing/adult-hybrid/adult-hybrid-Race_White.csv",
        "../datasets/preprocessing/adult-one-hot/adult-one-hot-Race_Amer-Indian-Eskimo.csv",
        "../datasets/preprocessing/adult-one-hot/adult-one-hot-Race_Asian-Pac-Islander.csv",
        "../datasets/preprocessing/adult-one-hot/adult-one-hot-Race_Black.csv",
        "../datasets/preprocessing/adult-one-hot/adult-one-hot-Race_Other.csv",
        "../datasets/preprocessing/adult-one-hot/adult-one-hot-Race_White.csv"]
    
    for f in files:
        print("## file: " + f)
        data, target = extractDataTarget(f)
        classifyLogisticRegression(data, target)
        classifyLogisticRegressionMultinomial(data, target)
        classifyHistGradientBoostingClassifier(data, target)
        classifySVC(data, target)
        classifyDecisionTreeClassifier(data, target)
        classifyKNeighborsClassifier(data, target)
        classifyNaiveBayesClassifier(data, target)
        classifyNNMLPClassifier(data, target)

    print()
    print("### Sex as proxy target")
    print()
    files = ["../datasets/preprocessing/adult-all-cat/adult-all-cat-Sex.csv",
        "../datasets/preprocessing/adult-hybrid/adult-hybrid-Sex_Female.csv",
        "../datasets/preprocessing/adult-hybrid/adult-hybrid-Sex_Male.csv",
        "../datasets/preprocessing/adult-one-hot/adult-one-hot-Sex_Female.csv",
        "../datasets/preprocessing/adult-one-hot/adult-one-hot-Sex_Male.csv"]
    
    for f in files:
        print("## file: " + f)
        data, target = extractDataTarget(f)
        classifyLogisticRegression(data, target)
        classifyLogisticRegressionMultinomial(data, target)
        classifyHistGradientBoostingClassifier(data, target)
        classifySVC(data, target)
        classifyDecisionTreeClassifier(data, target)
        classifyKNeighborsClassifier(data, target)
        classifyNaiveBayesClassifier(data, target)
        classifyNNMLPClassifier(data, target)


    print("========================= End Adult Dataset =========================")
    print()


def processCompas():

    print("==================== Preprocessing Compas Dataset ====================")
    print()

    print("### Race as proxy target")
    print()
    files = ["../datasets/preprocessing/compas/compas-African_American.csv",
        "../datasets/preprocessing/compas/compas-Asian.csv",
        "../datasets/preprocessing/compas/compas-Hispanic.csv",
        "../datasets/preprocessing/compas/compas-Native_American.csv",
        "../datasets/preprocessing/compas/compas-Other.csv"]
    
    for f in files:
        print("## file: " + f)
        data, target = extractDataTarget(f)
        classifyLogisticRegression(data, target)
        classifyLogisticRegressionMultinomial(data, target)
        classifyHistGradientBoostingClassifier(data, target)
        classifySVC(data, target)
        classifyDecisionTreeClassifier(data, target)
        classifyKNeighborsClassifier(data, target)
        classifyNaiveBayesClassifier(data, target)
        classifyNNMLPClassifier(data, target)

    print()
    print("### Sex as proxy target")
    print()
    files = ["../datasets/preprocessing/compas/compas-Female.csv"]
    
    for f in files:
        print("## file: " + f)
        data, target = extractDataTarget(f)
        classifyLogisticRegression(data, target)
        classifyLogisticRegressionMultinomial(data, target)
        classifyHistGradientBoostingClassifier(data, target)
        classifySVC(data, target)
        classifyDecisionTreeClassifier(data, target)
        classifyKNeighborsClassifier(data, target)
        classifyNaiveBayesClassifier(data, target)
        classifyNNMLPClassifier(data, target)

    print("========================= End Compas Dataset =========================")
    print()


def processGerman():

    print("==================== Preprocessing German Dataset ====================")
    print()

    print("### age_cat as proxy target")
    print()
    files = ["../datasets/preprocessing/german-all-cat/german-all-cat-age_cat.csv",
        "../datasets/preprocessing/german-hybrid/german-hybrid-age_cat.csv",
        "../datasets/preprocessing/german-one-hot/german-one-hot-age_cat.csv"]
    
    for f in files:
        print("## file: " + f)
        data, target = extractDataTarget(f)
        classifyLogisticRegression(data, target)
        classifyLogisticRegressionMultinomial(data, target)
        classifyHistGradientBoostingClassifier(data, target)
        classifySVC(data, target)
        classifyDecisionTreeClassifier(data, target)
        classifyKNeighborsClassifier(data, target)
        classifyNaiveBayesClassifier(data, target)
        classifyNNMLPClassifier(data, target)

    print()
    print("### age as proxy target")
    print()
    files = ["../datasets/preprocessing/german-all-cat/german-all-cat-age.csv",
        "../datasets/preprocessing/german-hybrid/german-hybrid-age.csv",
        "../datasets/preprocessing/german-one-hot/german-one-hot-age.csv"]
    
    for f in files:
        print("## file: " + f)
        data, target = extractDataTarget(f)
        classifyLogisticRegression(data, target)
        classifyLogisticRegressionMultinomial(data, target)
        classifyHistGradientBoostingClassifier(data, target)
        classifySVC(data, target)
        classifyDecisionTreeClassifier(data, target)
        classifyKNeighborsClassifier(data, target)
        classifyNaiveBayesClassifier(data, target)
        classifyNNMLPClassifier(data, target)


    print()
    print("### foreign_worker as proxy target")
    print()
    files = ["../datasets/preprocessing/german-all-cat/german-all-cat-foreign_worker.csv",
        "../datasets/preprocessing/german-hybrid/german-hybrid-foreign_worker.csv",
        "../datasets/preprocessing/german-one-hot/german-one-hot-foreign_worker.csv"]
    
    for f in files:
        print("## file: " + f)
        data, target = extractDataTarget(f)
        classifyLogisticRegression(data, target)
        classifyLogisticRegressionMultinomial(data, target)
        classifyHistGradientBoostingClassifier(data, target)
        classifySVC(data, target)
        classifyDecisionTreeClassifier(data, target)
        classifyKNeighborsClassifier(data, target)
        classifyNaiveBayesClassifier(data, target)
        classifyNNMLPClassifier(data, target)

    print("========================= End German Dataset =========================")
    print()


def processTitanic():

    print("==================== Preprocessing Titanic Dataset ====================")
    print()

    print("### Age as proxy target")
    print()
    files = ["../datasets/preprocessing/titanic-no-name/titanic-no-name-Age.csv"]
    
    for f in files:
        print("## file: " + f)
        data, target = extractDataTarget(f)
        target = target.astype(int)
        classifyLogisticRegression(data, target)
        classifyLogisticRegressionMultinomial(data, target)
        classifyHistGradientBoostingClassifier(data, target)
        classifySVC(data, target)
        classifyDecisionTreeClassifier(data, target)
        classifyKNeighborsClassifier(data, target)
        classifyNaiveBayesClassifier(data, target)
        classifyNNMLPClassifier(data, target)

    print()
    print("### Sex as proxy target")
    print()
    files = ["../datasets/preprocessing/titanic-no-name/titanic-no-name-Sex.csv"]
    
    for f in files:
        print("## file: " + f)
        data, target = extractDataTarget(f)
        classifyLogisticRegression(data, target)
        classifyLogisticRegressionMultinomial(data, target)
        classifyHistGradientBoostingClassifier(data, target)
        classifySVC(data, target)
        classifyDecisionTreeClassifier(data, target)
        classifyKNeighborsClassifier(data, target)
        classifyNaiveBayesClassifier(data, target)
        classifyNNMLPClassifier(data, target)


    print("========================= End Titanic Dataset =========================")
    print()


def processRicci():

    print("==================== Preprocessing Ricci Dataset ====================")
    print()

    print("### Race as proxy target")
    print()
    files = ["../datasets/preprocessing/ricci-all-cat/ricci-all-cat-Race.csv",
    "../datasets/preprocessing/ricci-one-hot/ricci-one-hot-Race_B.csv",
    "../datasets/preprocessing/ricci-one-hot/ricci-one-hot-Race_H.csv",
    "../datasets/preprocessing/ricci-one-hot/ricci-one-hot-Race_W.csv"]
    
    for f in files:
        print("## file: " + f)
        data, target = extractDataTarget(f)
        classifyLogisticRegression(data, target)
        classifyLogisticRegressionMultinomial(data, target)
        classifyHistGradientBoostingClassifier(data, target)
        classifySVC(data, target)
        classifyDecisionTreeClassifier(data, target)
        classifyKNeighborsClassifier(data, target)
        classifyNaiveBayesClassifier(data, target)
        classifyNNMLPClassifier(data, target)


    print("========================= End Ricci Dataset =========================")
    print()



#==============================================================================

if __name__ == '__main__':
    #processAdult()
    #processCompas()
    #processGerman()
    #processTitanic()
    processRicci()



