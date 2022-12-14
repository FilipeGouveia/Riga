#-------------------------------------------------------------------------------------
#
#   Author: Filipe Gouveia
#   11/05/2022
#   
#   File that helps processing and cleaning the datasets
#
#-------------------------------------------------------------------------------------

import argparse
import shutil
import os.path
from os.path import exists
import pandas as pd
import numpy as np

def joinFiles(filenames, outfile, copyfile):
    print("\tJoining files...\n")
    with open(outfile, 'w') as of:
        for fname in filenames:
            with open(fname) as infile:
                for line in infile:
                    of.write(line)
    shutil.copyfile(outfile, copyfile)

def dropEmptyEntries(dataframe, by_column=0):
    print("\tRemoving lines with empty values...\n")
    nan_value = float("NaN")
    dataframe.replace("?", nan_value, inplace=True)
    dataframe.dropna(inplace=True, axis=by_column)

def removeColumn(dataframe, column):
    print("\tRemoving column " + column + "...\n")
    dataframe = dataframe.drop([column], axis=1)
    return dataframe

#-------------------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',\
        choices=['adult', 'bank-marketing', 'communities', 'compas', 'credit', 'diabetes', 'dutch', 'german', 'KDD', 'lawschool', 'OULAD', 'ricci', 'students'],\
        help='Choose the dataset to process.', required=True)
    parser.add_argument('--complete', action='store_true', help='Join the information of the dataset in one file.')
    parser.add_argument('--filter_empty', action='store_true', help='Remove lines with empty values.')
    parser.add_argument('--remove_column', help='Remove a given column from the dataset.')
    parser.add_argument('--categorize_target', action='store_true', help='Transform target class in categorical values.')
    parser.add_argument('--copy', action='store_true', help='Makes a copy of the original to workfile.')


    args = parser.parse_args()

    if args.dataset == 'adult':
        print("=== Processing Adult dataset ===")
        print()
        filenames = ['Adult/adult-labels.txt', 'Adult/adult.data','Adult/adult.test']
        outfile_complete = 'Adult/Adult-complete.csv'
        workfile = 'Adult/Adult.csv'

        if args.complete:
            joinFiles(filenames, outfile_complete, workfile)

        if not os.path.exists(workfile):
            print("ERROR: Make sure you have file \"" + workfile + "\" available.")
            exit()

        df = pd.read_csv(workfile, skipinitialspace = True)
        df.columns = df.columns.str.lstrip(" ")
        df['target']=df['target'].str.strip(to_strip='.')

        if args.filter_empty:
            dropEmptyEntries(df)
        
        if args.remove_column:
            df = removeColumn(df, args.remove_column)

        df.to_csv(workfile, index=False)
        

    elif args.dataset == 'bank-marketing':
        print("=== Processing Bank-Marketing dataset ===")
    
    elif args.dataset == 'communities':
        print("=== Processing Communities and Crime dataset ===")
        print()

        filenames = ['communities and crime/communities-labels.txt', 'communities and crime/communities.data']
        outfile_complete = 'communities and crime/communities-complete.csv'
        workfile = 'communities and crime/communities.csv'

        if args.complete:
            joinFiles(filenames, outfile_complete, workfile)

        if not os.path.exists(workfile):
            print("ERROR: Make sure you have file \"" + workfile + "\" available.")
            exit()  

        df = pd.read_csv(workfile, skipinitialspace = True)


        if args.categorize_target:
            df['ViolentCrimesPerPop'] = np.where(df['ViolentCrimesPerPop'] > 0.7, "high-crime", "low-crime")

        if args.filter_empty:
            dropEmptyEntries(df, 1)

        df.to_csv(workfile, index=False)

    elif args.dataset == 'compas':
        print("=== Processing COMPAS dataset ===")
        print()
        
        #workfile = 'Cleaned/Compas/Compas.csv'
        workfile = 'Cleaned/Compas/Compas-within30days.csv'

        if args.copy:
            shutil.copyfile('Original/Compas/compas-scores-two-years.csv', workfile)

        if not os.path.exists(workfile):
            print("ERROR: Make sure you have file \"" + workfile + "\" available.")
            exit()

        df = pd.read_csv(workfile, skipinitialspace = True)

        if args.remove_column:
            df = removeColumn(df, args.remove_column)

        if args.filter_empty:
            dropEmptyEntries(df)         

        df.to_csv(workfile, index=False)

    elif args.dataset == 'credit':
        print("=== Processing Credit Card Clients dataset ===")

    elif args.dataset == 'diabetes':
        print("=== Processing Diabetes dataset ===")

        workfile = 'diabetes/diabetes.csv'

        if args.copy:
            shutil.copyfile('diabetes/diabetic_data.csv', workfile)

        if not os.path.exists(workfile):
            print("ERROR: Make sure you have file \"" + workfile + "\" available.")
            exit()

        df = pd.read_csv(workfile, skipinitialspace = True)

        if args.filter_empty:
            dropEmptyEntries(df)

        if args.remove_column:
            df = removeColumn(df, args.remove_column)

        df.to_csv(workfile, index=False)


    elif args.dataset == 'dutch':
        print("=== Processing Dutch Census dataset ===")

    elif args.dataset == 'german':
        print("=== Processing German-credit dataset ===")

    elif args.dataset == 'KDD':
        print("=== Processing KDD Census-Income dataset ===")
        print()

        filenames = ['KDD Census-Income/census-income-labels.txt', 'KDD Census-Income/census-income.data','KDD Census-Income/census-income.test']
        outfile_complete = 'KDD Census-Income/census-income-complete.csv'
        workfile = 'KDD Census-Income/census-income.csv'

        if args.complete:
            joinFiles(filenames, outfile_complete, workfile)

        if not os.path.exists(workfile):
            print("ERROR: Make sure you have file \"" + workfile + "\" available.")
            exit()

        df = pd.read_csv(workfile, skipinitialspace = True)
        df.columns = df.columns.str.lstrip(" ")
        df['income']=df['income'].str.strip(to_strip='.')

        if args.remove_column:
            df = removeColumn(df, args.remove_column)

        if args.filter_empty:
            dropEmptyEntries(df)         

        df.to_csv(workfile, index=False)

    elif args.dataset == 'lawschool':
        print("=== Processing Lawschool dataset ===")
        print()

        workfile = 'lawschool/lsac.csv'

        if not os.path.exists(workfile):
            print("ERROR: Make sure you have file \"" + workfile + "\" available.")
            exit()

        df = pd.read_csv(workfile, skipinitialspace = True)

        if args.remove_column:
            df = removeColumn(df, args.remove_column)

        if args.filter_empty:
            dropEmptyEntries(df)         

        df.to_csv(workfile, index=False)


    elif args.dataset == 'OULAD':
        print("=== Processing OULAD dataset ===")
        print()
        
        workfile = 'OULAD/OULAD.csv'

        if args.copy:
            shutil.copyfile('OULAD/studentInfo.csv', workfile)

        if not os.path.exists(workfile):
            print("ERROR: Make sure you have file \"" + workfile + "\" available.")
            exit()

        df = pd.read_csv(workfile, skipinitialspace = True)

        if args.remove_column:
            df = removeColumn(df, args.remove_column)

        if args.filter_empty:
            dropEmptyEntries(df)         

        df.to_csv(workfile, index=False)


    elif args.dataset == 'ricci':
        print("=== Processing Ricci dataset ===")
    
    elif args.dataset == 'students':
        print("=== Processing Students Performance dataset ===")

    else:
        parser.print_help()



