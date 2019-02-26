#-----------------------------------------------------------------------------
# CS 244 Assignment 3
# Team 20: Christine Wang, Yi-Huei Ho, Jiawei Chiang, Xinyun Zou
#
# Creation Date : Tue 24 Oct 2017 11:34:39 AM PDT
# Last Modified : Tue 31 Oct 2017 01:40:39 PM PDT
#----------------------------------------------------------------------------- 

import numpy as np
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt
import numpy
import scipy as sc
from scipy.interpolate import interp1d
import pandas as pd
import math
from decimal import *

def butter_bandpass(lowcut, highcut, fs, order=1):
    '''
    This function calculates butter bandpass
    '''
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=1):
    '''
    This function constructs bandpass filtered signal
    '''
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def calculate_HR(dataset, fs=50):
    '''
    This function calculates real-time heart rate from given IR readings
    '''
    dataset_HR = butter_bandpass_filter(dataset.IR, 1.0, 2.0, fs, order=1)
    plt.figure(1)
    plt.plot(dataset_HR)
    plt.title("Bandpass-Filtered 1st-Order IR Signal for HR\n (low_cut = 1.0, high_cut = 2.0)")
    #plt.show()

    #Calculate moving average with 0.75s in both directions, then append do dataset
    hrw = 0.75 #One-sided window size, as proportion of the sampling frequency
    mov_avg = pd.rolling_mean(dataset_HR, window=int(hrw*fs)) #Calculate moving average
    #Impute where moving average function returns NaN, which is the beginning of the signal where x hrw
    avg_hr = (np.mean(dataset_HR))
    mov_avg = [avg_hr if math.isnan(x) else x for x in mov_avg]
    mov_avg = [x*1.2 for x in mov_avg] #For now we raise the average by 20% to prevent the secondary heart contraction from interfering, in part 2 we will do this dynamically
    dataset['hart_rollingmean'] = mov_avg #Append the moving average to the dataframe
    #Mark regions of interest
    window = []
    peaklist = []
    newpeaklist = []
    newybeat = []
    listpos = 0 #We use a counter to move over the different data columns
    for datapoint in dataset_HR:
        rollingmean = dataset.hart_rollingmean[listpos] #Get local mean
        if (datapoint < rollingmean) and (len(window) < 1): #If no detectable R-complex activity -> do nothing
            listpos += 1

        elif (datapoint > rollingmean): #If signal comes above local mean, mark ROI
            window.append(datapoint)
            listpos += 1
            
        else: #If signal drops below local mean -> determine highest point
            maximum = max(window)
            beatposition = listpos - len(window) + (window.index(max(window))) #Notate the position of the point on the X-axis
            peaklist.append(beatposition) #Add detected peak to list
            window = [] #Clear marked ROI
            listpos += 1
    ybeat = [dataset_HR[x] for x in peaklist] #Get the y-value of all peaks for plotting purposes

    for i in range(len(peaklist)):
        if (ybeat[i]>62 and peaklist[i]>450):
            newpeaklist.append(peaklist[i])
            newybeat.append(ybeat[i])
    plt.figure(2)
    plt.title("Detected Peaks in IR Signal for Heart Rate")
    #plt.xlim(2600,2800)
    plt.ylim(-600,600)
    plt.plot(dataset_HR, alpha=0.5, color='blue') #Plot semi-transparent HR
    plt.plot(mov_avg, color ='green') #Plot moving average
    plt.scatter(newpeaklist, newybeat, color='red') #Plot detected peaks
    #plt.show()

    hr_interval= []
    cnt = 0
    heart_rate = []
    while (cnt < (len(newpeaklist)-1)):
        RR_interval = (newpeaklist[cnt+1] - newpeaklist[cnt]) #Calculate distance between beats in # of samples
        ms_dist = ((1.0*RR_interval / 50) * 1000.0) #Convert sample distances to ms distances
        hr_interval.append(ms_dist) #Append to list
        cnt += 1
    for i in range(len(hr_interval)):
        heart_rate.append(60000/hr_interval[i])
    print('heart rate (beats/min) = ',heart_rate)
    hr_peaktime = [dataset.time[x] for x in newpeaklist] # peak time for heart rate
    plt.figure(3)
    plt.plot(heart_rate)
    plt.title("Plot of Heart Rate")
    plt.xlabel("Peaks")
    plt.ylabel("HR (beats/min)")
    #plt.show()

    return (heart_rate, hr_peaktime)


def calculate_RR(dataset, fs=50):
    '''
    This function calculates real-time respiration rate from given IR readings
    '''
    dataset_RR = butter_bandpass_filter(dataset.IR, 0.2, 0.667, fs, order=5)
    plt.figure(4)
    plt.plot(dataset_RR)
    plt.title("Bandpass-Filtered 5th-Order IR Signal for RR\n (low_cut = 0.2, high_cut = 0.667)")
    #plt.show()

    #Calculate moving average with 0.75s in both directions, then append do dataset
    hrw = 0.75 #One-sided window size, as proportion of the sampling frequency
    mov_avg = pd.rolling_mean(dataset_RR, window=int(hrw*fs))
    #Impute where moving average function returns NaN, which is the beginning of the signal where x hrw
    avg_hr = (np.mean(dataset_RR))
    mov_avg = [avg_hr if math.isnan(x) else x for x in mov_avg]
    mov_avg = [x*1.2 for x in mov_avg] #For now we raise the average by 20% to prevent the secondary heart contraction from interfering, in part 2 we will do this dynamically
    dataset['hart_rollingmean'] = mov_avg #Append the moving average to the dataframe
    #Mark regions of interest
    window = []
    peaklist = []
    newpeaklist = []
    newybeat = []
    listpos = 0 #We use a counter to move over the different data columns
    for datapoint in dataset_RR:
        rollingmean = dataset.hart_rollingmean[listpos] #Get local mean
        if (datapoint < rollingmean) and (len(window) < 1): #If no detectable R-complex activity -> do nothing
            listpos += 1

        elif (datapoint > rollingmean): #If signal comes above local mean, mark ROI
            window.append(datapoint)
            listpos += 1
            
        else: #If signal drops below local mean -> determine highest point
            maximum = max(window)
            beatposition = listpos - len(window) + (window.index(max(window))) #Notate the position of the point on the X-axis
            peaklist.append(beatposition) #Add detected peak to list
            window = [] #Clear marked ROI
            listpos += 1
    ybeat = [dataset_RR[x] for x in peaklist] #Get the y-value of all peaks for plotting purposes
    rr_peaktime = [dataset.time[x] for x in peaklist] # peak time for respiration rate
    rr_interval = []
    cnt = 0
    respiration_rate = []
    while (cnt < (len(peaklist)-1)):
        RR_interval = (peaklist[cnt+1] - peaklist[cnt]) #Calculate distance between beats in # of samples
        ms_dist = ((1.0*RR_interval / 50) * 1000.0) #Convert sample distances to ms distances
        rr_interval.append(ms_dist) #Append to list
        cnt += 1
    for i in range(len(rr_interval)):
        respiration_rate.append(60000/rr_interval[i])
    respiration_rate = respiration_rate[1:]
    rr_peaktime = rr_peaktime[1:]
    print('respiration rate (breaths/min) = ',respiration_rate)

    plt.figure(5)
    plt.title("Detected Peaks in IR Signal for Respiration Rate")
    plt.plot(dataset_RR, alpha=0.5, color='blue') #Plot semi-transparent HR
    plt.plot(mov_avg, color ='green') #Plot moving average
    plt.scatter(peaklist, ybeat, color='red') #Plot detected peaks
    #plt.show()

    plt.figure(6)
    plt.plot(respiration_rate)
    plt.title("Plot of Respiration Rate")
    plt.xlabel("Peaks")
    plt.ylabel("RR (breaths/min)")
    #plt.show()

    return (respiration_rate, rr_peaktime)


def calculate_SPO2(dataset, fs=50):
    '''
    This function calculates real-time SPO2 from given IR and RED readings
    '''
    dataset_IR = butter_bandpass_filter(dataset.IR, 1.0, 2.0, fs, order=5)
    plt.figure(7)
    plt.plot(dataset_IR)
    plt.title("Bandpass-Filtered 5th-Order IR Signal for SPO2\n (low_cut = 1.0, high_cut = 2.0)")
    #plt.show()

    #Calculate moving average with 0.75s in both directions, then append do dataset
    hrw = 0.75 #One-sided window size, as proportion of the sampling frequency
    mov_avg = pd.rolling_mean(dataset_IR, window=int(hrw*fs)) #Calculate moving average

    #Impute where moving average function returns NaN, which is the beginning of the signal where x hrw
    avg_hr = (np.mean(dataset_IR))
    mov_avg = [avg_hr if math.isnan(x) else x for x in mov_avg]
    mov_avg = [x*1.2 for x in mov_avg] #For now we raise the average by 20% to prevent the secondary heart contraction from interfering, in part 2 we will do this dynamically
    dataset['hart_rollingmean'] = mov_avg #Append the moving average to the dataframe
    #Mark regions of interest
    window = []
    peaklist = []
    newpeaklist = []
    IR_peak = []
    listpos = 0 #We use a counter to move over the different data columns
    for datapoint in dataset_IR:
        rollingmean = dataset.hart_rollingmean[listpos] #Get local mean
        if (datapoint < rollingmean) and (len(window) < 1): #If no detectable R-complex activity -> do nothing
            listpos += 1

        elif (datapoint > rollingmean): #If signal comes above local mean, mark ROI
            window.append(datapoint)
            listpos += 1
            
        else: #If signal drops below local mean -> determine highest point
            maximum = max(window)
            beatposition = listpos - len(window) + (window.index(max(window))) #Notate the position of the point on the X-axis
            #if (datapoint>150):
            peaklist.append(beatposition) #Add detected peak to list
            window = [] #Clear marked ROI
            listpos += 1
            
    IR_peak = [dataset_IR[x] for x in peaklist] 

    convertedIR = -dataset_IR;

    #Calculate moving average with 0.75s in both directions, then append do dataset
    hrw = 0.75 #One-sided window size, as proportion of the sampling frequency
    mov_avg = pd.rolling_mean(convertedIR, window=int(hrw*fs)) #Calculate moving average

    #Impute where moving average function returns NaN, which is the beginning of the signal where x hrw
    avg_hr = (np.mean(convertedIR))
    mov_avg = [avg_hr if math.isnan(x) else x for x in mov_avg]
    mov_avg = [x*1.2 for x in mov_avg] #For now we raise the average by 20% to prevent the secondary heart contraction from interfering, in part 2 we will do this dynamically
    dataset['hart_rollingmean'] = mov_avg #Append the moving average to the dataframe
    #Mark regions of interest    
        
    minlist = []
    newvalleylist = []
    IR_min = []    
    window = []    
    listpos = 0    
    for datapoint in convertedIR:
        rollingmean = dataset.hart_rollingmean[listpos] #Get local mean
        if (datapoint < rollingmean) and (len(window) < 1): #If no detectable R-complex activity -> do nothing
            listpos += 1
        elif (datapoint > rollingmean): #If signal comes above local mean, mark ROI
            window.append(datapoint)
            listpos += 1
        else: #If signal drops below local mean -> determine highest point
            maximum = max(window)
            beatposition = listpos - len(window) + (window.index(max(window))) #Notate the position of the point on the X-axis
            #if (datapoint>150):
            minlist.append(beatposition) #Add detected peak to list
            window = [] #Clear marked ROI
            listpos += 1

    IR_min = [-convertedIR[x] for x in minlist] 

    num=len(IR_min)
    DC_IR=[]
    AC_IR=[]
    for i in range(1, num) :
        x=np.linspace(minlist[i-1], minlist[i])
        y=np.linspace(IR_min[i-1], IR_min[i])
        f=interp1d(x,y)
        DC_IR.append(f(peaklist[i]))
      
    for i in range(1, num) :
        AC_IR.append(IR_peak[i]-DC_IR[i-1])
        
    newpeaklist=peaklist[1:]

    plt.figure(8)
    plt.title("Detected Peaks in IR Signal for SPO2")
    plt.ylim(-2000, 2000)
    plt.xlim(2000, 2500)
    plt.plot(dataset_IR)
    plt.scatter(peaklist, IR_peak, color='r')
    plt.scatter(minlist, IR_min, color='b')
    plt.scatter(newpeaklist, DC_IR,  color='g')
    #plt.show()
    
    dataset_RED = butter_bandpass_filter(dataset.RED, 1.0, 2.0, fs, order=5)
    plt.figure(9)
    plt.plot(dataset_RED)
    plt.title("Bandpass-Filtered 5th-Order RED Signal for SPO2\n (low_cut = 1.0, high_cut = 2.0)")
    #plt.show()

    hrw = 0.75 #One-sided window size, as proportion of the sampling frequency
    mov_avg = pd.rolling_mean(dataset_RED, window=int(hrw*fs)) #Calculate moving average

    #Impute where moving average function returns NaN, which is the beginning of the signal where x hrw
    avg_hr = (np.mean(dataset_RED))
    mov_avg = [avg_hr if math.isnan(x) else x for x in mov_avg]
    mov_avg = [x*1.2 for x in mov_avg] #For now we raise the average by 20% to prevent the secondary heart contraction from interfering, in part 2 we will do this dynamically
    dataset['hart_rollingmean'] = mov_avg #Append the moving average to the dataframe
    #Mark regions of interest
    window = []
    peaklist = []
    newpeaklist = []
    RED_peak = []
    listpos = 0 #We use a counter to move over the different data columns
    for datapoint in dataset_RED:
        rollingmean = dataset.hart_rollingmean[listpos] #Get local mean
        if (datapoint < rollingmean) and (len(window) < 1): #If no detectable R-complex activity -> do nothing
            listpos += 1

        elif (datapoint > rollingmean): #If signal comes above local mean, mark ROI
            window.append(datapoint)
            listpos += 1
            
        else: #If signal drops below local mean -> determine highest point
            maximum = max(window)
            beatposition = listpos - len(window) + (window.index(max(window))) #Notate the position of the point on the X-axis
            #if (datapoint>150):
            peaklist.append(beatposition) #Add detected peak to list
            window = [] #Clear marked ROI
            listpos += 1
            
    RED_peak = [dataset_RED[x] for x in peaklist] 
    convertedRED = -dataset_RED;

    #Calculate moving average with 0.75s in both directions, then append do dataset
    hrw = 0.75 #One-sided window size, as proportion of the sampling frequency
    mov_avg = pd.rolling_mean(convertedRED, window=int(hrw*fs)) #Calculate moving average

    #Impute where moving average function returns NaN, which is the beginning of the signal where x hrw
    avg_hr = (np.mean(convertedRED))
    mov_avg = [avg_hr if math.isnan(x) else x for x in mov_avg]
    mov_avg = [x*1.2 for x in mov_avg] #For now we raise the average by 20% to prevent the secondary heart contraction from interfering, in part 2 we will do this dynamically
    dataset['hart_rollingmean'] = mov_avg #Append the moving average to the dataframe
    #Mark regions of interest    
        
    minlist = []
    newvalleylist = []
    RED_min = []    
    window = []    
    listpos = 0    
    for datapoint in convertedRED:
        rollingmean = dataset.hart_rollingmean[listpos] #Get local mean
        if (datapoint < rollingmean) and (len(window) < 1): #If no detectable R-complex activity -> do nothing
            listpos += 1

        elif (datapoint > rollingmean): #If signal comes above local mean, mark ROI
            window.append(datapoint)
            listpos += 1
            
        else: #If signal drops below local mean -> determine highest point
            maximum = max(window)
            beatposition = listpos - len(window) + (window.index(max(window))) #Notate the position of the point on the X-axis
            #if (datapoint>150):
            minlist.append(beatposition) #Add detected peak to list
            window = [] #Clear marked ROI
            listpos += 1

    RED_min = [-convertedRED[x] for x in minlist] 

    num=len(RED_min)
    DC_RED=[]
    AC_RED=[]
    for i in range(1, num) :
        x=np.linspace(minlist[i-1], minlist[i])
        y=np.linspace(RED_min[i-1], RED_min[i])
        f=interp1d(x,y)
        DC_RED.append(f(peaklist[i]))
      
    for i in range(1, num) :
        AC_RED.append(RED_peak[i]-DC_RED[i-1])

    plt.figure(10)
    plt.title("Detected Peaks in RED Signal for SPO2")
    plt.plot(dataset_RED, 'y')
    plt.scatter(peaklist, RED_peak, color='r')
    #plt.show()

    spo2=[]
    for i in range(len(DC_IR)):
        r=(AC_IR[i]*DC_IR[i])/(DC_RED[i]*AC_RED[i])
        spo2.append(-45.060*r*r + 30.354 *r + 94.845)

    spo2_peaktime = [dataset.time[x] for x in minlist]

    spo2 = spo2[5:]
    spo2_peaktime = spo2_peaktime[5:]

    print('SPO2 (%) = ',spo2)
    plt.figure(11)
    plt.plot(spo2)
    plt.title("Plot of SPO2")
    plt.xlabel("Peaks")
    plt.ylabel("SPO2 (%)")
    #plt.show()
    
    return (spo2, spo2_peaktime)
    

def find_ind(myList,myValue):
    '''
    This function finds the index of the first element in myList that is
    equal to or larger than myValue
    '''
    final_ind = -1
    for i,v in enumerate(myList):
        if v < myValue:
            final_ind = i
        else:
            break
    final_ind += 1

    return final_ind

    
def align_data(time, rate, peaktime):
    '''
    This function aligns the rate to the time steps according to the peak time
    '''
    peak_idx = []
    for pt in peaktime:
        peak_idx.append(find_ind(time,pt))
    aligned_rate = [None] * len(time)
    aligned_rate[:peak_idx[0]] = [rate[0]]*peak_idx[0]
    aligned_rate[peak_idx[-1]:] = [rate[-1]]*(len(aligned_rate)-peak_idx[-1])
    for i, rt in enumerate(rate):
        aligned_rate[peak_idx[i]:peak_idx[i+1]]=[rt]*(peak_idx[i+1]-peak_idx[i])
    
    return aligned_rate


if __name__ == '__main__':
    # load the original dataset
    dataset = pd.read_csv("takashin_Homework_sample.csv")
    numpy.set_printoptions(threshold=numpy.nan)
    # calculate the heart rate and align to the original time steps
    heart_rate, hr_peaktime = calculate_HR(dataset)
    aligned_hr = align_data(dataset.time.tolist(), heart_rate, hr_peaktime)
    # calculate the respiration rate and align to the original time steps
    respiration_rate, rr_peaktime = calculate_RR(dataset)
    aligned_rr = align_data(dataset.time.tolist(), respiration_rate, rr_peaktime)
    # calculate the SPO2 and align to the original time steps
    spo2, spo2_peaktime = calculate_SPO2(dataset)
    aligned_spo2 = align_data(dataset.time.tolist(), spo2, spo2_peaktime)
    # write IR, RED, HR, RR, SPO2 values to a new csv file
    aligned_hr = [float(Decimal("%.3f" % hr)) for hr in aligned_hr]
    aligned_rr = [float(Decimal("%.3f" % rr)) for rr in aligned_rr]
    aligned_spo2 = [float(Decimal("%.3f" % sp)) for sp in aligned_spo2]
    HR = pd.Series(data=aligned_hr,name='Heart Rate (beats/min)')#,dtype='int64')
    RR = pd.Series(data=aligned_rr,name='Respiration Rate (breaths/min)')#,dtype='int64')
    SPO2 = pd.Series(data=aligned_spo2,name='SPO2 (%)')#,dtype='int64')
    col0 = dataset.IR.reset_index(drop=True)
    col1 = dataset.RED.reset_index(drop=True)
    col2 = HR.reset_index(drop=True)
    col3 = RR.reset_index(drop=True)
    col4 = SPO2.reset_index(drop=True)
    All_to_Write = pd.concat([col0,col1,col2,col3,col4],axis=1)
    All_to_Write.to_csv('team20_assignment3.csv', index = False, header = True)
    # show all the plots in the functions called earlier
    plt.show()

