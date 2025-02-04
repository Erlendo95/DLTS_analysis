import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.constants import h, k, m_e, elementary_charge
import math
import random

#Smoothing function based on moving average
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='same')  

#Function to find Mean Square Error
def MSE(data,fit):
    cMSE = 0
    for i in range(len(data)):
        cMSE += (data[i]-fit[i])**2
    cMSE = cMSE/len(data)
    return(cMSE/np.mean(data))

#reads in a DLTS-file
def readData(fileName):
    f = open(fileName, 'r')
    f.readline()
    f.readline()
    f.readline()
    T = [] #K
    W = [] #dC/C
    C = [] #pF
   
    for line in f:
        try:
            splitLine = line.split(' ')
            T.append(float(splitLine[0]))
            C.append(float(splitLine[1]))
            W.append(splitLine[12:17]) #Set to 3:9 for lockin for IDEFIX, TIFFY or Asterix. Set to 5:11 for Lockin from Obelix and 12:17 for GS4 from Obelix
        except:
            continue
    
    return(T,W,C)

#returns the window of interest from a list of DLTS-windows
def convertWindow(W,n):
    cW = np.zeros(len(W))
    
    for i in range(len(W)-1):
        cW[i] = float(W[i][n])
        
    return(cW)
    
#Plots all the DLTS-windows
def plotAll(T,W):
    for i in range(len(W[0])):
        cW = convertWindow(W,i)
        plt.plot(T,cW, label = "W"+str(i+1), linewidth = 5)
    plt.legend()
    plt.show()

#Plots the DLTS-window of interest
def plotOne(T,W,nW, mode):
    if(mode == "DLTS"):
        
        #cW = convertWindow(W,nW)
        plt.plot(T,W, label = "W"+str(nW+1), linewidth = 4)
        plt.legend()
    if(mode == "PL"):
        plt.plot(T,W,label = str(mode))
    

#creates a single Gauss function
def createSingleGauss(T, amp,cen,wid):
    return(amp * np.exp(-(T - cen)**2 / (2 * wid**2)))

#Creates a generalized fit based of any number of Gaussians used in the fit
def Gaussing(T, *param):
    fit = 0
    for i in range(int(len(param)/3)):
        fit = fit+(param[i*3] * np.exp(-(T - param[i*3+1])**2 / (2 * param[i*3+2]**2)))
    return(fit)
  
#function used to remoe NaN from the datafile  
def cleanNaN(W,T):
    cW = []
    cT = []
    for i in range(len(W)):
        if(math.isnan(W[i])):
            print(W[i])
        else:
            cW.append(W[i])
            cT.append(T[i])
    
    return(cT,cW)

#Normalization by division. Not only way to normalize, but easiest in this context.
#Also multiply by 10 to simplify fitting
def normalize(y):
    maximum = np.max(y)
    minimum = np.min(y)
    sc = maximum-minimum
    for i in range(len(y)):
        y[i] = (y[i])/(maximum)*10
    return(y,sc/10,minimum)

#Returns text in the format xEy
def format_e(n):
    a = '%E' % n
    return a.split('E')[0].rstrip('0').rstrip('.') + 'E' + a.split('E')[1]            

#Write out data on the form, Temperature, Raw data, Fit, Gauss1, Gauss2, ..., GaussN
def writeToFile(T,popt,W,pGauss):
    f = open('FitSHow.txt', 'w')
    FIT = Gaussing(T,*popt)
    
    for i in range(len(T)):
        f.write(str(T[i]) + " " + str(W[i]) + " " + str(FIT[i]) + " " + str(pGauss[0][i]) + " " + str(pGauss[1][i]) + " " + str(pGauss[2][i]) + " " + str (pGauss[3][i]) + " \n")
    f.close()
        
    
#Boltzmann
kb = 8.617E-5

#Smoothing function based on moving average
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='same')        

from scipy.ndimage import median_filter

#Fitting function
def fitData(T,W,nW,cwl,sF):    
    mx = 10000000
    sc = 1
    mi = 0
    pGauss = []
    nGauss = 4 #number of gaussians to be fitted
   
    W,sc,mi = normalize(W)
    
    
    #Initial guesses on the form [Amplitude, Temperature, width] 
    Gauss1 = [4,420,500] #gauss curve 1
    Gauss2 = [7,450,5]
    Gauss3 = [7,620,5]
    Gauss4 = [9,60020,5]
    Gauss5 = [6,500,5]
    Gauss6 = [4,50,5]
    Background = [0.2,400,30]
    gaussParam = np.concatenate((Gauss1,Gauss2)) #adds all initial gauss guesses together. The amount of Gauss-curves must be filled in
    lower_bounds = [0.4] * int(len(gaussParam)/3) * 3
    upper_bounds = [np.inf] *int(len(gaussParam)/3 )* 3
    popt, pcov = curve_fit(Gaussing, T, W, p0=gaussParam, maxfev=mx, bounds=(lower_bounds,upper_bounds)) #makes fit
    plt.plot(T, Gaussing(T, *popt), color='red', linewidth = 4, linestyle = '--') #plotfit of whole dataset
    plt.plot(T,W, label =  nW)
    plt.title("#3(900)")
    
    #creates individual gauss curve list for plotting
    wn = 1
    for i in range(0,int(len(gaussParam)/3)):
            amp = (popt[i*3])
            cen = popt[i*3+1]
            wid = popt[i*3+2]
            pGauss.append(createSingleGauss(T,amp,cen,wid))
            print("  ")
            print("Gauss curve " + str(i))
            print("Amplitude: " + str(amp))
            print("Center wavelength: " + str(cen) )
            print("Width: " + str(wid))
            wn = wn+1
            #plots each individual gauss curve        
    for j in range(int(len(gaussParam)/3)):
            plt.plot(T,pGauss[j], '--', label =j+1, linewidth = 1)
           
            plt.xlabel('Temperature (K)', fontsize = 15)
            plt.ylabel('DLTS signal (dC/C)',fontsize = 15)
            plt.xticks(fontsize = 15)
            plt.yticks(fontsize = 15)
            plt.legend(loc = "upper left")
    
    #writeToFile(T,popt,W,pGauss) #uncomment if writing to file
    
    return(popt,sc,mi)
        
#Finding index of a temperature
def findTemp(T,temp):
    for i in range(len(T)):
        if T[i] == temp:
            return(i)
        
#Flattens a spectrum       
def flatten(W):
    diff = 0
    if(W[0]<W[-1]):
        diff = W[0]
    else:
        diff = W[-1]
    
    for i in range(len(W)):
        W[i] = W[i]-diff
    
    return(W)        


fileName = "C:/Users/erlendou/OneDrive - Universitetet i Oslo/Desktop/PhD/Paper4/Divakans/Obelix/#3(900)_DLTS_300to650_10V_(600um)_50ms_20window.txt" #Filepath  
      
T,W,C = readData(fileName)

peaks = []
r_mi = 1 #window to be fitted, with 0 being the shortest time window
r_ma = r_mi+1   
#plot a certain window
for wn in range(r_mi,r_ma):       
    nW = wn #number of the DLTS window to be plotted
    cT = 0
    uW = convertWindow(W, nW)
    
    #Smoothing functions using either moving average of a median filter
    #uW = median_filter(uW, size=4)
    #uW = moving_average(uW, 5)
    
    xMin = findTemp(T,440) #Set minimum temperature of the window
    xMax = findTemp(T,570) #Set maximum temperature of the window
    T=T[xMin:xMax]
    
    uW = uW[xMin:xMax]
   
    peak,sc,mi = fitData(T,uW,nW+1,cT,1)
    peaks.append(peak)


ampP = []
cenP = []
widP = []
nT = []
nTGauss = []
eRate = [1.44896/0.02,1.81717/0.04,2.09799/0.08,2.28229/0.16,2.39056/0.32,2.44976/0.64]
eRateGS4 = [70.66836,38.64953,20.28329,10.40057,5.267648]
eRate = eRateGS4
tl = [0.02,0.04,0.08,0.16,0.32,0.64]
F = [0.12488, 0.15490, 0.17597, 0.18880, 0.19594, 0.19971]

eRate = eRate[r_mi-1:r_ma-1]
tl = tl[r_mi:r_ma]
F = F[r_mi-1:r_ma-1]

winP = []
th = 50
ind = 0

qPeaks = []

for i in range(int(len(peaks[0])/3)):
    qPeaks.append(peaks[0][1+i*3])
    ampP.append(peaks[0][0+i*3]*sc)

#Prints data to be used in DLTS_arrhenius.py
print(qPeaks)
print("Temperatures for peak :" +str(r_ma) + ": ")
print(np.sort(qPeaks))
print(ampP)

