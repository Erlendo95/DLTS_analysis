# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 15:47:56 2025

@author: erlendou
"""
#Import the normal stuff
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import h, k, m_e, elementary_charge

#Write numbers on the from xEy
def format_e(n):
    a = '%E' % n
    return a.split('E')[0].rstrip('0').rstrip('.') + 'E' + a.split('E')[1] 

#Get unceirtanty based on spread of fit
def uncertainty(coeff,cov, y, Ea, ccs):
    da = coeff[0]-cov[0]
    db = abs(coeff[1]-cov[1])
    
    dEa = Ea-(-da*k/elementary_charge) 
    dccs = ccs*db #Progressive unceirtanty
    
    return(dEa, dccs)

#Difference between mean of trap concentration and largest deviators
def nDiff(nT):
    diff = abs(nT[0]-np.mean(nT))
    for n in nT:
        if(abs(n-np.mean(nT)) > diff):
            diff = abs(n-np.mean(nT))
    return(diff)


eRate = [1.44896/0.02,1.81717/0.04,2.09799/0.08,2.28229/0.16,2.39056/0.32,2.44976/0.64] #Emission rate of lockin windows
eRateGS4 = [70.66836,38.64953,20.28329,10.40057,5.267648] #Emission rate of GS4 windows

F = [0.12488, 0.15490, 0.17597, 0.18880, 0.19594, 0.19971] #Window constant used for calculating nT
y = (2**(5/2)*3**(1/2)*np.pi**(3/2)*k**2*0.2*m_e/h**3) #Material constant used for calculating ccs

T_raw = [] #List of peak temperatures, designed to be a list of lists containing all peaks in each window
A_raw = [] #List of amplitudes, must have same dimension as T_raw

T_list_tot = [[]] #List of lists, where the number of lists must match the amount of peaks fitted
A_list_tot = [[]] #Same as T_list_tot

nP = len(T_list_tot) #Number of peaks to be fitted

#Sorting the peaks from the raw data 
for i in range(int(len(T_raw)/nP)):
    for j in range(nP):
        T_list_tot[j].append(T_raw[j+i*nP])
        A_list_tot[j].append(A_raw[j+i*nP])


ai = 0 #index used for nT-calculation

#Loop calculating the parameters for each peak
for T_list in T_list_tot:
    arr = np.zeros(len(T_list))
    TP = np.zeros(len(T_list))
    Amean = np.mean(A_list_tot[ai])
   
    ai = ai+1
   
    nD = 6.60E+16 #Carrier concentration at RT
    cb = 56 #reverse bias capacitance at RT
    
    nT = []
    
    for i in range(len(arr)):
        
        arr[i] = np.log(eRateGS4[i]/T_list[i]**2)
        TP[i] = 1./T_list[i]
        nT.append(2*Amean*nD/(cb*F[i]))
        
    coeffs, cov = np.polyfit(TP, arr, 1, cov = True)
    a = coeffs[0]
    b = coeffs[1]
    fit = a*TP + b
    Ea = -a*k/elementary_charge
    ccs = 1E4*np.exp(b)/y
    errors = np.sqrt(np.diag(cov))
    uncertainties = uncertainty(coeffs, errors, y, Ea, ccs)
    plt.scatter(TP, arr)
    plt.plot(TP,fit, '--')
        
    print("Activation energy of peak at " + str(np.mean(T_list)) + " is " + str(Ea) + "  ±" + str(uncertainties[0]))
    print("Capture cross section is " + str(format_e(ccs)) + "  ±" + str(format_e(uncertainties[1])) )
    print("Concentration is " + str(format_e(np.mean(nT))) + "  ±" + str(format_e(nDiff(nT))) + "\n")
    
    