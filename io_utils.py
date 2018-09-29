#!/usr/bin/env python
# -*- coding: utf-8 -*-


import pandas as pd
from sys import argv


def read_input(filename):
    return pd.read_csv(filename, delimiter=',')


if __name__ == "__main__":
    df = read_input(argv[1])
    print(df)
