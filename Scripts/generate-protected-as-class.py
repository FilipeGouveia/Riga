#-------------------------------------------------------------------------------------
#
#   Author: Filipe Gouveia
#   31/05/2022
#   
#   Script to generate the versions of a dataset without the original classification
#   target and with each protected attribute as new classification target
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



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', help='Dataset file to process.', required=True)
    parser.add_argument('-o', '--outfolder', help='Folder of the output files.', required=True)
    parser.add_argument('-c', '--classification', help='Class atribute to be removed.', required=True)
    parser.add_argument('-p', '--protected', nargs='+', help='List of protected attributes to generate dataset versions with each protected attribute as class.')

    args = parser.parse_args()

    os.makedirs(args.outfolder, exist_ok=True)

    filebasename = Path(args.dataset).stem

    df = pd.read_csv(args.dataset, skipinitialspace = True)

    df = df.drop([args.classification], axis=1)

    non_protected_columns = [col for col in df.columns if col not in args.protected]

    for label in args.protected:
        headers = non_protected_columns.copy()
        headers.append(label)
        outputfilename = filebasename + "-" + label + ".csv"
        output = os.path.join(args.outfolder, outputfilename)
        df.to_csv(output, columns=headers, index = False)




