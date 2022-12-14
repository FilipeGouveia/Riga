#-------------------------------------------------------------------------------------
#
#   Author: Filipe Gouveia
#   20/06/2022 (last modified: 14/11/2022)
#   
#   Script to identify potential proxy attributes of a given dataset.
#   It searches for proxy attributes in Decision Trees with an iterative approach by
#   considering increasing values for the trees maximum depth. 
#   It can be parameterized with an impurity threshold and maximum depth.
#   Options to draw (some) trees can be provided.
#   Options to output the proxies found as text can be provided.
#
#-------------------------------------------------------------------------------------

import argparse
from ensurepip import bootstrap
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
from copy import deepcopy
from joblib import dump, load

def get_no_impurity_depth(tree, impurity_threshold = 0.0, text = False, featureNames = None):

    result_depth = []
    min_depth = None
    max_samples_min_depth = None
    max_samples = None
    min_depth_max_samples = None

    n_nodes = tree.node_count
    children_left = tree.children_left
    children_right = tree.children_right
    impurities = tree.impurity
    samples = tree.n_node_samples
    parents = np.zeros(shape=n_nodes, dtype=np.int64)
    result_nodes = []

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
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
            parents[children_left[node_id]] = node_id
            parents[children_right[node_id]] = node_id
        #else:
        if impurities[node_id] <= impurity_threshold :
            result_depth.append(depth)
            result_nodes.append(node_id)
            if((min_depth is None) or depth < min_depth):
                min_depth = depth
                max_samples_min_depth = samples[node_id]
            elif(depth == min_depth and samples[node_id] > max_samples_min_depth):
                max_samples_min_depth = samples[node_id]

            if((max_samples is None) or samples[node_id] > max_samples):
                max_samples = samples[node_id]
                min_depth_max_samples = depth
            elif(samples[node_id] == max_samples and depth < min_depth_max_samples):
                min_depth_max_samples = depth

    if text:
        print_node_text(tree, result_nodes, parents, children_left, children_right, featureNames)

    return result_depth, min_depth, max_samples_min_depth, max_samples, min_depth_max_samples

def print_node_text(tree, target_nodes, parents, children_left, children_right, featureNames):
    
    for node in target_nodes:

        p_str = ""
        node_p = node
        while node_p != 0:
            parent = parents[node_p]
            aux = ""
            if featureNames is not None :
                aux += str(featureNames[tree.feature[parent]])
            else :
                aux += str(tree.feature[parent])
            if node_p == children_left[parent]:
                aux = aux + " <= "
            else:
                aux = aux + " > "
            aux = aux + str(tree.threshold[parent])
            p_str = aux + " , " + p_str
            node_p = parent
        print("\t** {proxies} #impurity: {impurity}, #samples: {samples}.\n".format(proxies=p_str, impurity=tree.impurity[node], samples=tree.n_node_samples[node]))

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', help='Dataset file to process.', required=True)
    parser.add_argument('-o', '--outfolder', help='Folder of the output files.', required=True)
    parser.add_argument('-c', '--classification', help='Class atribute to be removed.', required=True)
    parser.add_argument('-t', '--type', choices=['decisiontree', 'randomforest', 'extratrees'], default="decisiontree")
    parser.add_argument('--drawdepth', type=int, help='Draw trees of chosen depth.')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-m', '--maxdepth', type=int, default=100, help='Max depth to search.')
    parser.add_argument('-i', '--impurity', type=float, help='Impurity threshold [0,1]. Default = 0.0', default=0.0)
    parser.add_argument('--sampleDraw', type=float, default=0.1, help='Percentage of samples in nodes with impurity lower than the threshold that we want to draw [0,1] Default: 0.1')
    parser.add_argument('--drawall', action='store_true')
    parser.add_argument('--drawNoImpurity', action='store_true')
    parser.add_argument('--text', action='store_true')


    args = parser.parse_args()
    drawdepth = None
    if args.drawdepth:
        drawdepth = args.drawdepth
    textOut = args.text

    os.makedirs(args.outfolder, exist_ok=True)

    filebasename = Path(args.dataset).stem

    global clf

    df = pd.read_csv(args.dataset, skipinitialspace = True)

    y = df[args.classification]
    X = df.drop([args.classification], axis=1)

    # one-hotencoding
    df_onehot = pd.get_dummies(X, prefix_sep="_", dtype=bool)

    classnames = y.unique()
    classnames.sort()
    classnames2 = [str(x) for x in classnames]

    #data_train, data_test, target_train, target_test = train_test_split(df_onehot, y, random_state=42)

    noImpurity = []
    #noImpurityByDepth = np.zeros(shape=len(df_onehot.columns))
    outfile = filebasename + "-" + args.type + ".txt"
    outputfile = os.path.join(args.outfolder, outfile)

    min_node_depth = None
    max_sample_min_depth = None
    best_model_min_depth = None
    max_node_sample = None
    min_depth_max_sample = None
    best_model_max_sample = None

    with open(outputfile, 'w') as f:

        print("File: " + args.dataset)
        print("This dataset has " + str(len(df_onehot.columns)) + " features after onehotencoding.", file=f)
        print("This dataset has " + str(len(df_onehot.columns)) + " features after onehotencoding.")


        for depth in range(1,(min(len(df_onehot.columns),args.maxdepth))+1):
        #for depth in range(1,(len(df_onehot.columns))+1):

            localNoImpurity = []

            outputfilebasename = filebasename + "-" + args.type + "-d" + str(depth)

            if args.type == 'decisiontree':
                if args.verbose:
                    print("\tDecision Tree Classifier with max depth " + str(depth), file=f)
                clf = DecisionTreeClassifier(max_depth=depth)
            elif args.type == 'randomforest':
                if args.verbose:
                    print("\tRandom Forest Classifier with max depth " + str(depth), file=f)
                clf = RandomForestClassifier(max_depth=depth, bootstrap=False, random_state=42)
            else:
                if args.verbose:
                    print("\tExtra Trees Classifier with max depth " + str(depth), file=f)
                clf = ExtraTreesClassifier(max_depth=depth)

            #_ = clf.fit(data_train, target_train)
            _ = clf.fit(df_onehot, y)

            #score = clf.score(data_test, target_test)

            #print("\t\tClassifier score: " + f"{score:.3f}", file=f)

            #cv_results = cross_validate(clf, df_onehot, y, cv=5)
            #scores = cv_results["test_score"]
            #print("\t\tThe mean cross-validation accuracy is: "
                #f"{scores.mean():.3f} +/- {scores.std():.3f}", file=f)

            if args.type == 'decisiontree':

                terminalNodes = get_no_impurity_depth(clf.tree_)
                noImpurity.extend(terminalNodes)
                localNoImpurity.extend(terminalNodes)
                if args.verbose:
                    print("\t\t" + str(terminalNodes)[1:-1], file=f)

                if drawdepth == depth:
                    outdraw = outputfilebasename + ".dot"
                    outputdraw = os.path.join(args.outfolder, outdraw)

                    dot_data = tree.export_graphviz(clf, out_file = outputdraw, feature_names=df_onehot.columns, class_names=classnames2, filled=True)
                    #$ dot -Tps tree.dot -o tree.ps      (PostScript format)
                    #$ dot -Tpng tree.dot -o tree.png    (PNG format)

            else:
                #draw the first 5 estimators only if depth == 4
                if drawdepth == depth:
                    drawrange = 5
                    if(args.drawall):
                        drawrange = len(clf.estimators_)
                    for index in range(0,drawrange):
                        outdraw = outputfilebasename + "-" + str(index) + ".dot"
                        outputdraw = os.path.join(args.outfolder, outdraw)

                        dot_data = tree.export_graphviz(clf.estimators_[index], out_file = outputdraw, feature_names=df_onehot.columns, class_names=classnames2, filled=True)
                        #$ dot -Tps tree.dot -o tree.ps      (PostScript format)
                        #$ dot -Tpng tree.dot -o tree.png    (PNG format)
                for index in range(0,len(clf.estimators_)):
                    terminalNodes, mindepth, maxsamplesmindepth, maxsamples, mindepthmaxsamples = get_no_impurity_depth(clf.estimators_[index].tree_, args.impurity, textOut, df_onehot.columns)
                    noImpurity.extend(terminalNodes)
                    localNoImpurity.extend(terminalNodes)

                    if (mindepth is not None) and (maxsamplesmindepth is not None):
                        if (min_node_depth is None) or mindepth < min_node_depth:
                            min_node_depth = mindepth
                            max_sample_min_depth = maxsamplesmindepth
                            best_model_min_depth = deepcopy(clf.estimators_[index])
                        elif mindepth == min_node_depth and maxsamplesmindepth > max_sample_min_depth:
                            max_sample_min_depth = maxsamplesmindepth
                            best_model_min_depth = deepcopy(clf.estimators_[index])
                    
                    if (maxsamples is not None) and (mindepthmaxsamples is not None):
                        if (max_node_sample is None) or maxsamples > max_node_sample:
                            max_node_sample = maxsamples
                            min_depth_max_sample = mindepthmaxsamples
                            best_model_max_sample = deepcopy(clf.estimators_[index])
                        elif maxsamples == max_node_sample and mindepthmaxsamples < min_depth_max_sample:
                            min_depth_max_sample = mindepthmaxsamples
                            best_model_max_sample = deepcopy(clf.estimators_[index])

                        percentage = maxsamples / len(df_onehot.index)
                        # print trees with node with zero (or threshold) impurity with more than threshold samples
                        if percentage >= args.sampleDraw:
                            outdraw = outputfilebasename + "-" + str(index) + ".dot"
                            outputdraw = os.path.join(args.outfolder, outdraw)
                            dot_data = tree.export_graphviz(clf.estimators_[index], out_file = outputdraw, feature_names=df_onehot.columns, class_names=classnames2, filled=True)
                        if args.drawNoImpurity and (mindepth is not None):
                            outdraw = outputfilebasename + "-" + str(index) + ".dot"
                            outputdraw = os.path.join(args.outfolder, outdraw)
                            dot_data = tree.export_graphviz(clf.estimators_[index], out_file = outputdraw, feature_names=df_onehot.columns, class_names=classnames2, filled=True)


                    if args.verbose:
                        print("\t\t(#" + str(index) + ")" + str(terminalNodes)[1:-1], file=f)

            print("\t\tDepth " + str(depth), file=f)
            print("\t\tDepth " + str(depth) + " with " + str(len(localNoImpurity)) + " nodes found.")
            print("\t\t\tList: " + str(localNoImpurity)[1:-1], file=f)
            print("\t\t\tNo Impurity found with depth of Average "
                f"{np.mean(localNoImpurity):.3f} and Median {np.median(localNoImpurity):.3f}", file=f)

        print("\n",file=f)
        print("\tList: " + str(noImpurity)[1:-1], file=f)
        print("\tNo Impurity found with depth of Average "
            f"{np.mean(noImpurity):.3f} and Median {np.median(noImpurity):.3f}", file=f)
        print("\tNo Impurity found with depth of Average "
            f"{np.mean(noImpurity):.3f} and Median {np.median(noImpurity):.3f}")


    if(best_model_min_depth is not None):
        dumpfile = filebasename + "-bestmodelmindepth.joblib"
        outputdumpfile = os.path.join(args.outfolder, dumpfile)
        dump(best_model_min_depth, outputdumpfile)

        print("\tBest model (min depth) found with a node at depth " + str(min_node_depth) + " with " + str(max_sample_min_depth) + " samples.")

        outdraw = filebasename + "-bestmodelmindepth.dot"
        outputdraw = os.path.join(args.outfolder, outdraw)
        dot_data = tree.export_graphviz(best_model_min_depth, out_file = outputdraw, feature_names=df_onehot.columns, class_names=classnames2, filled=True)
        if textOut:
            get_no_impurity_depth(best_model_min_depth.tree_, args.impurity, textOut, df_onehot.columns)

    if(best_model_max_sample is not None):
        dumpfile = filebasename + "-bestmodelmaxsample.joblib"
        outputdumpfile = os.path.join(args.outfolder, dumpfile)
        dump(best_model_max_sample, outputdumpfile)

        print("\tBest model (max sample) found with " + str(max_node_sample) + " sample in a node at depth " + str(min_depth_max_sample) + ".")

        outdraw = filebasename + "-bestmodelmaxsample.dot"
        outputdraw = os.path.join(args.outfolder, outdraw)
        dot_data = tree.export_graphviz(best_model_max_sample, out_file = outputdraw, feature_names=df_onehot.columns, class_names=classnames2, filled=True)
        if textOut:
            get_no_impurity_depth(best_model_max_sample.tree_, args.impurity, textOut, df_onehot.columns)

    print()

    