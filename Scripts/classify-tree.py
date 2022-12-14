#-------------------------------------------------------------------------------------
#
#   Author: Filipe Gouveia
#   31/05/2022 (last modified: 20/06/2022)
#   
#   Script to test different decision tree-based classifiers to train a model to
#   predict a protected attribute, checking the accuracy. High accuracy is an
#   indicator of the presence of proxy attributes
#
#-------------------------------------------------------------------------------------

import argparse
import shutil
import os
import os.path
from os.path import exists
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import tree
from matplotlib import pyplot as plt



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', help='Dataset file to process.', required=True)
    parser.add_argument('-o', '--outfolder', help='Folder of the output files.', required=True)
    parser.add_argument('-c', '--classification', help='Class atribute to be removed.', required=True)
    parser.add_argument('-t', '--type', choices=['decisiontree', 'randomforest', 'extratrees'], default="decisiontree")
    parser.add_argument('--depth', type=int, help='List of protected attributes to generate dataset versions with each protected attribute as class.')

    args = parser.parse_args()

    os.makedirs(args.outfolder, exist_ok=True)

    filebasename = Path(args.dataset).stem

    outputfilebasename = filebasename + "-" + args.type

    outfile = outputfilebasename + ".txt"
    outputfile = os.path.join(args.outfolder, outfile)

    depth = None
    if args.depth:
        depth = args.depth

    with open(outputfile, 'w') as f:

        df = pd.read_csv(args.dataset, skipinitialspace = True)

        y = df[args.classification]
        X = df.drop([args.classification], axis=1)

        # one-hotencoding
        df_onehot = pd.get_dummies(X, prefix_sep="_", dtype=bool)

        data_train, data_test, target_train, target_test = train_test_split(df_onehot, y, random_state=42)

        global clf

        
        if args.type == 'decisiontree':
            print("\tDecision Tree Classifier", file=f)
            clf = DecisionTreeClassifier(max_depth=depth)
        elif args.type == 'randomforest':
            print("\tRandom Forest Classifier", file=f)
            clf = RandomForestClassifier(max_depth=depth)
        else:
            print("\tExtra Trees Classifier", file=f)
            clf = ExtraTreesClassifier(max_depth=depth)
        
        _ = clf.fit(data_train, target_train)

        score = clf.score(data_test, target_test)

        print("\t\tClassifier score: " + f"{score:.3f}", file=f)

        cv_results = cross_validate(clf, df_onehot, y, cv=5)
        scores = cv_results["test_score"]
        print("\t\tThe mean cross-validation accuracy is: "
            f"{scores.mean():.3f} +/- {scores.std():.3f}", file=f)

    
        classnames = y.unique()
        classnames.sort()
        classnames2 = [str(x) for x in classnames]
        if args.type == 'decisiontree':
            outdraw = outputfilebasename + ".dot"
            outputdraw = os.path.join(args.outfolder, outdraw)

            dot_data = tree.export_graphviz(clf, out_file = outputdraw, feature_names=df_onehot.columns, class_names=classnames2, filled=True)
            #$ dot -Tps tree.dot -o tree.ps      (PostScript format)
            #$ dot -Tpng tree.dot -o tree.png    (PNG format)

        else:
            #draw the first 5 estimators
            for index in range(0,5):
                outdraw = outputfilebasename + "-" + str(index) + ".dot"
                outputdraw = os.path.join(args.outfolder, outdraw)

                dot_data = tree.export_graphviz(clf.estimators_[index], out_file = outputdraw, feature_names=df_onehot.columns, class_names=classnames2, filled=True)
                #$ dot -Tps tree.dot -o tree.ps      (PostScript format)
                #$ dot -Tpng tree.dot -o tree.png    (PNG format)

        