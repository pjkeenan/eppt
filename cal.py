import os
import uproot3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import sys

cwd = os.getcwd()

def get_df(path, filename):
    print("Getting file: " + filename + "..." )

    try:
        file = uproot3.open(path+filename)["B5"]
    except FileNotFoundError:
        print("Wrong file or file path")
        return -1
    else:
        try:
            keys = file.keys()
            keys_decode = [key.decode('ASCII') for key in keys]
            dataframe = file.pandas.df(keys, flatten=False)
            dataframe.columns = keys_decode
        except KeyError:
            print("Key Error")
            return -1

    print("done...")

    return dataframe

def index_to_grid(index):
    # convert index to a row, column on a grid
    row = index % 4
    column = (index - row) / 4
    return (int(row), int(column))

def create_data_dict(df):
    """Extract data
	
	Arguments
	---------
	df : dataframe, list or similar
		data in extracted via uproot format
	Returns
	------
	dc1_dict : OrderedDict
		dictionary of data from DC1. Format is
        (evtNum : [[x1, x2, ..., xn], [z1, z2, ..., zn]])
    dc2_dict : OrderedDict
        dictionary of data from DC2. Format is
        (evtNum : [[x1, x2, ..., xn], [z1, z2, ..., zn]])
	"""
    df_EC = df["ECEnergyVector"].to_numpy()
    # df_dc1_z = df["Dc1HitsVector_z"].to_numpy()
    # df_dc2_x = df["Dc2HitsVector_x"].to_numpy()
    # df_dc2_z = df["Dc2HitsVector_z"].to_numpy()
    
    # print(f'df_EC: {df_EC}')

    # dicts to hold output data 
    EC_dict = OrderedDict()
    # dc2_dict = OrderedDict()
    
    for event in range(len(df_EC)): # this is loop over events
        # EC_cellList = []
        # EC_EnergyList = []
        # dc2_xlist = []
        # dc2_zlist = []
        EC_dict[event] = df_EC[event]
    
    # print(f'EC_dict: {EC_dict}')
    # print(f'\nEC_dict[980][40]: {EC_dict[980][40]}')
    # sys.exit()
    return(EC_dict)

def calc_clus_E(eList):
    # Return E of all clusters as a list?

    # Use BFS method to find cluster of cells with max EDep
    # Have 20 x 4 grid of cells
    Lx = 20
    Ly = 4
    # Direction vectors to get adj points
    #      u  d  l  r  ul  ur ll lr
    dr = [-1, 1, 0, 0, -1, -1, 1, 1]
    dc = [0, 0, -1, 1, -1, 1, -1, 1]
    
    # initalise grid and add energy dep in the cell
    grid = np.zeros((Ly, Lx))
    # print(grid)
    for i in range(len(eList)):
        r, c = index_to_grid(i)
        # E = eList[i]
        # print(f'E: {eList[i]}, r: {r} c: {c}')
        # print(grid[x, y])
        grid[r, c] = eList[i]
    print(grid)
    # visited = []
    clusEList = []
    nClus = 0
    # indexQ = []
    for i in range(len(eList)):
        # print('\n######################')
        rowQ = []
        colQ = []
        rowQ.append(index_to_grid(i)[0])
        colQ.append(index_to_grid(i)[1])
        
        # print(f'1 - rowQ: {rowQ} colQ: {colQ}')
        clusE = 0
        while rowQ:
            # dequeue an index 
            row = rowQ.pop(0)
            col = colQ.pop(0)
            # row, column = index_to_grid(index)
            # Case cell has deposited energy & not already visited 
            # visited cells marked w/ -ve values
            # print(f'2 - rowQ: {rowQ} colQ: {colQ}')
            if grid[row, col] > 0:
                for dRow, dCol in zip(dr, dc):
                    # print(f'dR: {dRow}, dC: {dCol}')
                    checkR = row + dRow
                    checkC = col + dCol
                    # Skip checks for positions out of grid
                    if checkR < 0 or checkR == Ly:
                        continue
                    elif checkC < 0 or checkC == Lx:
                        continue
                    # Found E dep in neighbour -> add cell to queue
                    elif grid[checkR, checkC] > 0:
                        # print('Found neighbour E dep')
                        rowQ.append(checkR)
                        colQ.append(checkC)
                        
                clusE += grid[row, col]
                print(f'E: {grid[row, col]} clusE = {clusE}')
                # set E dep value as -ve to mark visited cell
                grid[row, col] = -abs(grid[row, col])
                # print('Set E to -ve')
            # else:
            #     print(f'E NOT > 0 - Site not in cluster or visited\n{grid[row, col]}')
        
        if clusE > 0:
            print('clusE > 0 - added to clusEList')
            clusEList.append(clusE)
            nClus +=1

    print('Final grid')       
    for y in range(Ly):
        # print(f'y: {y}')
        for x in range(Lx):
            # print(f'x: {x}')
            print(f'{grid[y, x]:.1f}', end=' ')
        print('\n')
    
    print(f'nClus: {nClus}')
    return clusEList


def main():
    # fileList = ['100GeV_0_25T_mu.root', '100GeV_0_5T_mu.root', '100GeV_1T_mu.root', '200GeV_0_5T_mu.root',  '50GeV_0_5T_mu.root']
    # fileList = ['100GeV_0_25T_mu.root']
    fileList = ['small.root']
    # BFieldList = [0.25, 0.5, 1, 0.5, 0.5]
    BFieldList = [0.5]
    for file, BField in zip(fileList, BFieldList):
        print(f'\nPlotting {file}')
        name = file.split('.')[0]
        # Get data
        df = get_df(cwd, f"/{file}")
        EC_dict = create_data_dict(df)

        for event in EC_dict:
            clusEList = calc_clus_E(EC_dict[event])
            print(f'clusEList: {clusEList}')
            # sys.exit()


if __name__=='__main__':
    main()