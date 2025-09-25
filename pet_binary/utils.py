# -*- coding: utf-8 -*-
import numpy as np


def process_label(df, path):
    df['label'] = 2
    path = path.rstrip('.csv')
    path = f'{path}.txt'
    with open(path, 'r') as fr:
        lines = fr.readlines()
        for i, line in enumerate(lines[:-1]):
            line = line.rstrip('\n').split(',')
            j, l = int(line[0]), int(line[1])
            k = int(lines[i+1].split(',')[0]) - 1
            if k<j:
                print(k)
            df.loc[j:k, 'label'] = l

        line = lines[-1].rstrip('\n').split(',')
        j, l = int(line[0]), int(line[1])
        df.loc[j:, 'label'] = 1


def process_feature(df):
    df['ln_pts'] = np.log(df['pts']+1)
    df['ln_dyn'] = np.log(df['pts_dyn']+1)
    df['ln_sta'] = np.log(df['pts']-df['pts_dyn']+1)

    df['z_iqr'] = df['z_q3'] - df['z_q1']

    df['range'] = np.sqrt(np.square(df['x_c'])+np.square(df['y_c']))

    df['dx'] = df['x_c'].diff().fillna(0)
    df['dy'] = df['y_c'].diff().fillna(0)
    df['dist'] = np.sqrt(np.square(df['dx'])+np.square(df['dy']))

    df.reset_index(inplace=True)