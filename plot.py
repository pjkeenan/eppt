from ast import Or
from tkinter import E
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, OrderedDict
import sys
import math
import os
import uproot3
from itertools import combinations
import scipy.stats as stats


cwd = os.getcwd()

def get_df(path, filename):
    print("Getting file: " + filename + "..." )

    try:
        file = uproot3.open(path+filename + ".root")["B5"]
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

def create_df(df):
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
    df_dc1_x = df["Dc1HitsVector_x"].to_numpy()
    df_dc1_z = df["Dc1HitsVector_z"].to_numpy()
    df_dc2_x = df["Dc2HitsVector_x"].to_numpy()
    df_dc2_z = df["Dc2HitsVector_z"].to_numpy()

    # dicts to hold output data 
    dc1_dict = OrderedDict()
    dc2_dict = OrderedDict()
    
    for event in range(len(df_dc1_z)): # this is loop over events
        dc1_xlist = []
        dc1_zlist = []
        dc2_xlist = []
        dc2_zlist = []
        for x, z in zip(df_dc1_x[event], df_dc1_z[event]):
            dc1_xlist.append(x / 1000) # mm -> m
            dc1_zlist.append(z * 0.5) #layer -> m Could change this to real space values 
        dc1_dict[event] = [dc1_xlist, dc1_zlist]
        for x, z in zip(df_dc2_x[event], df_dc2_z[event]):
            dc2_xlist.append(x / 1000) # mm -> m
            dc2_zlist.append(z * 0.5) # layer -> m
        dc2_dict[event] = [dc2_xlist, dc2_zlist]
    
    return(dc1_dict, dc2_dict)

def fit_hits (df_ax1, df_ax2):
    """Perform linear least squares fit on data
       and calculate gradient and intercept of line       
	
	Arguments
	---------
	df_ax1 : dataframe, list or similar
		data for horizontal to fit
	df_ax2 : dataframe, list or similar
		data for vertical to fit
	Returns
	------
	slope : float
		gradient of line resulting from fit
    intercept : float
        intercept of line resulting from fit
	"""
    # get index of each occourance of each z value. Add to corresponding list
    layer0 = []
    layer1 = []
    layer2 = []
    layer3 = []
    layer4 = []
    for i in range(len(df_ax1)):
        if df_ax1[i] == 0:
            layer0.append(i)
        elif df_ax1[i] == 0.5:
            layer1.append(i)
        elif df_ax1[i] == 1.0:
            layer2.append(i)
        elif df_ax1[i] == 1.5:
            layer3.append(i)
        elif df_ax1[i] == 2.0:
            layer4.append(i)
        else:
            print(f"{df_ax1[i]} did not match - ERROR")
            sys.exit()

    print(f"l0: {layer0}")
    print(f"l1: {layer1}")
    print(f"l2: {layer2}")
    print(f"l3: {layer3}")
    print(f"l4: {layer4}")

    minChi2 = 999
    for l0 in layer0:
        for l1 in layer1:
            for l2 in layer2:
                for l3 in layer3:
                    for l4 in layer4:
                        xvals = [df_ax2[l0], df_ax2[l1], df_ax2[l2], df_ax2[l3], df_ax2[l4]]
                        zvals = [df_ax1[l0], df_ax1[l1], df_ax1[l2], df_ax1[l3], df_ax1[l4]]
                        chi2, p = stats.chisquare(xvals)
                        chi2 = abs(chi2)
                        # print(f"xvals: {xvals} chi2: {chi2}")
                        if chi2 < minChi2:
                            minChi2 = chi2
                            slope, intercept = np.polyfit(zvals, xvals, 1)
                            slope = abs(slope)
                            # intercept = intercept

    # slope, intercept = np.polyfit(df_ax1, df_ax2, 1)
    # slope = abs(slope) # converting to m
    # intercept = intercept
    return(slope, intercept)

def deflection_angle(m1, m2):
    theta = math.atan(m1+m2/(1+(m1*m2)))
    return(theta)

def calc_momentum(Bfield, length, m1, m2):
    theta = deflection_angle(m1, m2)
    mom = (0.3*Bfield*length)/(2*math.sin(theta/2)) # GeV????
    return(mom)

def calculate_binning(data):
    # Reset bin finding params
    nbins = 50
    lowestedge = 1000
    highestedge = -1000

    # data = pd.Series(data)

    # Find the bin range
    if min(data) < lowestedge:
        lowestedge = min(data)
        print(f"Changing lowestedge to: {lowestedge}")
    else:
        print("Not changing lowestegde")
    if max(data) > highestedge:
        highestedge = max(data)
        print(f"Changing highestedge to: {highestedge}")
    else:
        print("Not changing highestedge")

    # Set the bin edges
    binedges = []
    binCentres = []
    binwidth = (highestedge - lowestedge)/nbins
    for binIndex in range(nbins + 1):
        binedges.append(lowestedge + (binIndex * binwidth))
    for binIndex in binedges[:-1]:
        binCentres.append(binIndex+(binwidth/2))
    return(binedges, binCentres, binwidth)

def remove_outliers(momList):
    filteredList = []
    nRemoved = 0
    mean = np.mean(momList)
    stdDev = np.std(momList)
    print(f"mean: {mean} stdDev: {stdDev}")
    
    for m in momList:
        if m < mean+(1.5*stdDev) and m > mean-(1.5*stdDev):
            filteredList.append(m)
        else:
            nRemoved +=1
        
    print(f"HERE min: {min(filteredList)} max: {max(filteredList)} kept: {len(filteredList)} removed: {nRemoved}")
    return(filteredList)




"""
# slope_dc1, intercept_dc1 = np.polyfit(df_dc1_filtered['dc1Hits_z'], df_dc1_filtered['dc1Hits_x'], 1)
slope_dc1, intercept_dc1 = fit_hits(df_dc1_filtered['dc1Hits_z'], df_dc1_filtered['dc1Hits_x'])
line_values_dc1 = [slope_dc1 * i + intercept_dc1 for i in df_dc1_filtered['dc1Hits_z']]

# slope_dc2, intercept_dc2 = np.polyfit(df_dc2_filtered['dc2Hits_z'], df_dc2_filtered['dc2Hits_x'], 1)
slope_dc2, intercept_dc2 = fit_hits(df_dc2_filtered['dc2Hits_z'], df_dc2_filtered['dc2Hits_x'])
line_values_dc2 = [slope_dc2 * i + intercept_dc2 for i in df_dc2_filtered['dc2Hits_z']]
"""
# plt.figure(1)
# # plt.scatter(df_dc1['dc1Hits_z'], df_dc1['dc1Hits_x'])
# plt.plot(df_dc1['dc1Hits_z'], df_dc1['dc1Hits_x'], 'or')
# plt.plot(df_dc1['dc1Hits_z'], line_values, 'b')
# plt.savefig('test.png')
# # plt.show()


def main():
    df = get_df(cwd, "/B5")
    dc1_dict, dc2_dict = create_df(df)
    momentumList = []
    for event in dc1_dict:
        m_dc1, c_dc1 = fit_hits(dc1_dict[event][1], dc1_dict[event][0])
        m_dc2, c_dc2 = fit_hits(dc2_dict[event][1], dc2_dict[event][0])
        # print(f"m_dc1: {m_dc1}, m_dc2: {m_dc2}")
        momentum = calc_momentum(0.5, 2, m_dc1, m_dc2)
        # print(f"momentum: {momentum}")
        momentumList.append(momentum)
    
    filteredList = remove_outliers(momentumList)
    

    # best fit of data
    mu, sigma = stats.norm.fit(filteredList)
    binedges, binCentres, binwidth = calculate_binning(filteredList)
    plt.figure(1)
    n, bins, patches = plt.hist(filteredList, bins=binedges, histtype='step')
    # plt.close()
    # plt.figure(2)
    # plt.plot(binCentres, n, '.')
    # y = stats.norm.pdf( binCentres, mu, sigma)
    # l = plt.plot(binCentres, y*binwidth*len(filteredList), 'r--', linewidth=2)
    plt.title(f'mu = {mu}, sigma = {sigma}')
    # plt.yscale('log')
    plt.savefig(f'figures/momDist.png')
    plt.close()
    

if __name__=='__main__':
    main()