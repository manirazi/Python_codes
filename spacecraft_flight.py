import sys   #loading libraries
import numpy
import matplotlib.pyplot
import pandas as pd
import operator
import math
import csv
from numpy import genfromtxt
from scipy import interp

def interpolation(data,point):  #this subroutine is used to find the missing values 
                                #between a range of data points with known bounds
    number_of_columns=(data.shape)
    if number_of_columns[1] != 2: # not a proper data set dimension, so print help message
        print('Usage: number of the culmns for the filenames\n \
               must be only 2')
        return
    
    data=pd.DataFrame(data,columns=['x','f'])  #assigning column names to data
    data=data.dropna(axis=0,how='any')   #removing rows with missing values

    M=len(data.x)     #number of grid points M
    data=data.sort_values(by=['x','f'],ascending=[1,0])   #sorting the data
    
    if point<data.x.iloc[0] or point>data.x.iloc[M-1]: # not a proper data set dimension, so print help message
        print('Your desired x value is out of the range for this data set; please try again',data.x.iloc[0],point,data.x.iloc[M-1])
        return    
    
    for i in range(M-2,-1,-1):
        if point<=data.x.iloc[i+1] and point>=data.x.iloc[i]:
            c1=data.f.iloc[i]
            if i==0 or i==1:
                DF=(-22.0*data.f.iloc[i]+36.0*data.f.iloc[i+1]-18.0*data.f.iloc[i+2]+4.0*data.f.iloc[i+3])/                                              (-22.0*data.x.iloc[i]+36.0*data.x.iloc[i+1]-18.0*data.x.iloc[i+2]+4.0*data.x.iloc[i+3])
                S2=(data.f.iloc[i+1]-data.f.iloc[i])/(data.x.iloc[i+1]-data.x.iloc[i])
                if i==1:
                    S1=(data.f.iloc[i+1]-data.f.iloc[i])/(data.x.iloc[i+1]-data.x.iloc[i])
                else:
                    S1=(data.f.iloc[i]-data.f.iloc[i-1])/(data.x.iloc[i]-data.x.iloc[i-1])
                c2=DF
            else:
                if i==M-2:
                    DF=(22.0*data.f.iloc[i]-36.0*data.f.iloc[i-1]+18.0*data.f.iloc[i-2]-4.0*data.f.iloc[i-3])/ \
                    (22.0*data.x.iloc[i]-36.0*data.x.iloc[i-1]+18.0*data.x.iloc[i-2]-4.0*data.x.iloc[i-3])
                    S1=(data.f.iloc[i]-data.f.iloc[i-1])/(data.x.iloc[i]-data.x.iloc[i-1])
                    S2=(data.f.iloc[i+1]-data.f.iloc[i])/(data.x.iloc[i+1]-data.x.iloc[i])
                    c2=DF
                else:
                    DF=(-data.f.iloc[i+2]+8.0*data.f.iloc[i+1]-8.0*data.f.iloc[i-1]+data.f.iloc[i-2])/ \
                    (-data.x.iloc[i+2]+8.0*data.x.iloc[i+1]-8.0*data.x.iloc[i-1]+data.x.iloc[i-2])
                    S3=(data.f.iloc[i-1]-data.f.iloc[i-2])/(data.x.iloc[i-1]-data.x.iloc[i-2])
                    S1=(data.f.iloc[i]-data.f.iloc[i-1])/(data.x.iloc[i]-data.x.iloc[i-1])
                    S2=(data.f.iloc[i+1]-data.f.iloc[i])/(data.x.iloc[i+1]-data.x.iloc[i])
                    S4=(data.f.iloc[i+2]-data.f.iloc[i+1])/(data.x.iloc[i+2]-data.x.iloc[i+1])
                    h3=(data.x.iloc[i-1]-data.x.iloc[i-2])
                    h1=(data.x.iloc[i]-data.x.iloc[i-1])
                    h2=(data.x.iloc[i+1]-data.x.iloc[i])
                    h4=(data.x.iloc[i+2]-data.x.iloc[i+1])
                    P1=(S1*(2.0*h1+h3)-S3*h1)/(h3+h1)
                    P2=(S1*h2+S2*h1)/(h1+h2)
                    P3=(S2*(2.0*h2+h4)-S4*h2)/(h2+h4)
                    M0=3.0*min(abs(S3),abs(S2),abs(P2))
                    if P2>0.0 and P1>0.0 and S1-S3>0.0 and S2-S1>0.0:
                        M0=max(M0,1.5*min(abs(P2),abs(P1)))
                    if P2<0.0 and P1<0.0 and S1-S3<0.0 and S2-S1<0.0:
                        M0=max(M0,1.5*min(abs(P2),abs(P1)))
                    if P2<0.0 and P3<0.0 and S2-S1>0.0 and S4-S2>0.0:
                        M0=max(M0,1.5*min(abs(P2),abs(P3)))
                    if P2>0.0 and P3>0.0 and S2-S1<0.0 and S4-S2<0.0:
                        M0=max(M0,1.5*min(abs(P2),abs(P3)))
                    if DF<0.0:
                        sigma=-1.0
                    else:
                        if DF>0.0:
                            sigma=1.0
                        else:
                            sigma=0.0
                    if P2<0.0:
                        sigma2=-1.0
                    else:
                        if P2>0.0:
                            sigma2=1.0
                        else:
                            sigma2=0.0
                    if sigma==sigma2:
                        c2=sigma*min(abs(DF),M0)
                    else:
                        c2=0.0
            S=(data.f.iloc[i+1]-data.f.iloc[i])/(data.x.iloc[i+1]-data.x.iloc[i])
            if i==0:
                c3=(3.0*S-(-22.0*data.f.iloc[i+1]+36.0*data.f.iloc[i+2]-18.0*data.f.iloc[i+3]+4.0*data.f.iloc[i+4])/ \
                    (-22.0*data.x.iloc[i+1]+36.0*data.x.iloc[i+2]-18.0*data.x.iloc[i+3]+4.0*data.x.iloc[i+4])-2.0*c2)/ \
                (data.x.iloc[i+1]-data.x.iloc[i])
            else:
                if i==M-3 or i==M-2:
                    c3=(3.0*S-(22.0*data.f.iloc[i+1]-36.0*data.f.iloc[i]+18.0*data.f.iloc[i-1]-4.0*data.f.iloc[i-2])/  \
                        (22.0*data.x.iloc[i+1]-36.0*data.x.iloc[i]+18.0*data.x.iloc[i-1]-4.0*data.x.iloc[i-2])-2.0*c2)/ \
                    (data.x.iloc[i+1]-data.x.iloc[i])
                else:
                    DFP=(-data.f.iloc[i+3]+8.0*data.f.iloc[i+2]-8.0*data.f.iloc[i]+data.f.iloc[i-1])/ \
                    (-data.x.iloc[i+3]+8.0*data.x.iloc[i+2]-8.0*data.x.iloc[i]+data.x.iloc[i-1])
                    S3=(data.f.iloc[i]-data.f.iloc[i-1])/(data.x.iloc[i]-data.x.iloc[i-1])
                    S1=(data.f.iloc[i+1]-data.f.iloc[i])/(data.x.iloc[i+1]-data.x.iloc[i])
                    S2=(data.f.iloc[i+2]-data.f.iloc[i+1])/(data.x.iloc[i+2]-data.x.iloc[i+1])
                    S4=(data.f.iloc[i+3]-data.f.iloc[i+2])/(data.x.iloc[i+3]-data.x.iloc[i+2])
                    h3=(data.x.iloc[i]-data.x.iloc[i-1])
                    h1=(data.x.iloc[i+1]-data.x.iloc[i])
                    h2=(data.x.iloc[i+2]-data.x.iloc[i+1])
                    h4=(data.x.iloc[i+3]-data.x.iloc[i+2])
                    P1=(S1*(2.0*h1+h3)-S3*h1)/(h3+h1)
                    P2=(S1*h2+S2*h1)/(h1+h2)
                    P3=(S2*(2.0*h2+h4)-S4*h2)/(h2+h4)
                    M0=3.0*min(abs(S3),abs(S2),abs(P2))
                    if P2>0.0 and P1>0.0 and S1-S3>0.0 and S2-S1>0.0:
                        M0=max(M0,1.5*min(abs(P2),abs(P1)))
                    if P2<0.0 and P1<0.0 and S1-S3<0.0 and S2-S1<0.0:
                        M0=max(M0,1.5*min(abs(P2),abs(P1)))
                    if P2<0.0 and P3<0.0 and S2-S1>0.0 and S4-S2>0.0:
                        M0=max(M0,1.5*min(abs(P2),abs(P3)))
                    if P2>0.0 and P3>0.0 and S2-S1<0.0 and S4-S2<0.0:
                        M0=max(M0,1.5*min(abs(P2),abs(P3)))
                    if DFP<0.0:
                        sigma=-1.0
                    else:
                        if DFP>0.0:
                            sigma=1.0
                        else:
                            sigma=0.0
                    if P2<0.0:
                        sigma2=-1.0
                    else:
                        if P2>0.0:
                            sigma2=1.0
                        else:
                            sigma2=0.0
                    if sigma==sigma2:
                        DFP=sigma*min(abs(DFP),M0)
                    else:
                        FP=0.0
                    c3=(3.0*S-DFP-2.0*c2)/(data.x.iloc[i+1]-data.x.iloc[i])
            if i==0:
                c4=-(2.0*S-(-22.0*data.f.iloc[i+1]+36.0*data.f.iloc[i+2]-18.0*data.f.iloc[i+3]+4.0*data.f.iloc[i+4])/ \
                     (-22.0*data.x.iloc[i+1]+36.0*data.x.iloc[i+2]-18.0*data.x.iloc[i+3]+4.0*data.x.iloc[i+4])-c2)/ \
                     ((data.x.iloc[i+1]-data.x.iloc[i])*(data.x.iloc[i+1]-data.x.iloc[i]))
            else:
                if i==M-3 or i==M-2 or i==M-1:
                    c4=-(2.0*S-(22.0*data.f.iloc[i+1]-36.0*data.f.iloc[i]+18.0*data.f.iloc[i-1]-4.0*data.f.iloc[i-2])/ \
                         (22.0*data.x.iloc[i+1]-36.0*data.x.iloc[i]+18.0*data.x.iloc[i-1]-4.0*data.x.iloc[i-2])-c2)/ \
                         ((data.x.iloc[i+1]-data.x.iloc[i])*(data.x.iloc[i+1]-data.x.iloc[i]))
                else:
                    c4=-(2.0*S-DFP-c2)/((data.x.iloc[i+1]-data.x.iloc[i])*(data.x.iloc[i+1]-data.x.iloc[i]))
            Interpolated_value=c1+(point-data.x.iloc[i])*c2+((point-data.x.iloc[i])*(point-data.x.iloc[i]))*c3+ \
            ((point-data.x.iloc[i])*(point-data.x.iloc[i])*(point-data.x.iloc[i]))*c4
            return Interpolated_value  #outputing the stimated value for the missing data point
            fig=matplotlib.pyplot.figure(figsize=(10.0,3.0))
            axes1 = fig.add_subplot(1,3,1)
            axes1.set_ylabel('function')
            axes1.set_xlabel('x')
            matplotlib.pyplot.plot(data.x.iloc[0:M-1],data.f.iloc[0:M-1],'r^')  #ploting the interpolated data points among the data set
            matplotlib.pyplot.plot([point],[Interpolated_value],'go')
            matplotlib.pyplot.show()
            break

data=pd.read_csv('data.csv',sep =',',header=0,names=['Altitude','Density','Sound_speed','Mach','CD'])  #loading the datafile as
                                                                                                        #as a dataframe

                   #input values for the spacecraft
w=5000.0     #initial weight with fuel
V=-6000.0   #exhaust velocity
D=2.0  #base diameter
A=0.25*(math.acos(-1.0))*(D*D)  #base area
rho0=0.002378   #sea level density
g=32.16   #gravitational acceleration
m=w/g
DM=4000.0  #fuel mass
Time=30.0  #projected time for burning the fuel
DMT=(DM/g)/Time  #mass loss rate due to fuel consumption
                #extracting different sets of data from the data file as vectors
Mach=data.Mach    #Mach number
Mach=Mach.dropna(axis=0,how='any')  #removing rows with missing values at the end of the data set  'NaN'
CD=data.CD  #drag coefficient
CD=CD.dropna(axis=0,how='any')  #removing rows with missing values  'NaN'
Altitude=data.Altitude

Density=data.Density
Sound_speed=data.Sound_speed
dt=0.01      #temporal increment for computation
time=60.0   #time of operation
M1=len(Mach)
M2=len(Altitude)
M3=int(round(time/dt))+1
v=[0.0 for i in range(M3+1)]  #initialization of vectors of unknown variables
t=[0.0 for i in range(M3+1)]
y=[0.0 for i in range(M3+1)]
M=[m for i in range(M3+1)]
Mach_and_CD=pd.DataFrame({'x':Mach,'f':CD})   #forming separate dataframes for finiding the missing values
                                                #during the course of computation
Altitude_and_Density=pd.DataFrame({'x':Altitude,'f':Density})
Altitude_and_Sound_speed=pd.DataFrame({'x':Altitude,'f':Sound_speed})




i=0

while t[i]<time:        #method of solution of the model differential equation is
                           #the standard 4th Runge-Kutta
    if t[i]<30.0:

        if y[i]>=Altitude[0] and y[i]<=Altitude[M2-1]:      #choosing between finding the missing values 
                                                              #through interpolation or regression
            rho1=interpolation(Altitude_and_Density,y[i]) 
            a1=interpolation(Altitude_and_Sound_speed,y[i])
        else:    
            rho1=interp(y[i],Altitude,Density) 
            a1=interp(y[i],Altitude,Sound_speed) 

        Mach1=v[i]/a1
        if Mach1>=Mach[0] and Mach1<=Mach[M1-1]:         #choosing between finding the missing values 
                                                              #through interpolation or regression
            CD1=interpolation(Mach_and_CD,Mach1) 
        else:
            CD1=interp(Mach1,Mach,CD)
        P1=v[i]
        k1=-g-(0.5*A*(v[i]*v[i])*rho1*CD1/M[i])-(V/M[i])*DMT;
        L1=-DMT
        if y[i]+0.5*dt*P1>=Altitude[0] and y[i]+0.5*dt*P1<=Altitude[M2-1]:   
            rho1=interpolation(Altitude_and_Density,y[i]+0.5*dt*P1)
            a1=interpolation(Altitude_and_Sound_speed,y[i]+0.5*dt*P1)
        else:
            rho1=interp(y[i]+0.5*dt*P1,Altitude,Density) 
            a1=interp(y[i]+0.5*dt*P1,Altitude,Sound_speed) 
        Mach1=(v[i]+0.5*dt*k1)/a1
        if Mach1>=Mach[0] and Mach1<=Mach[M1-1]:
            CD1=interpolation(Mach_and_CD,Mach1) 
        else:
            CD1=interp(Mach1,Mach,CD) 
        P2=v[i]+0.5*dt*k1
        k2=-g-(0.5*A*((v[i]+0.5*dt*k1)*(v[i]+0.5*dt*k1))*rho1*CD1/(M[i]+0.5*dt*L1))-(V/(M[i]+0.5*dt*L1))*DMT
        L2=-DMT
        if y[i]+0.5*dt*P2>=Altitude[0] and y[i]+0.5*dt*P2<=Altitude[M2-1]:
            rho1=interpolation(Altitude_and_Density,y[i]+0.5*dt*P2)
            a1=interpolation(Altitude_and_Sound_speed,y[i]+0.5*dt*P2)
        else:
            rho1=interp(y[i]+0.5*dt*P2,Altitude,Density)
            a1=interp(y[i]+0.5*dt*P2,Altitude,Sound_speed) 
        Mach1=(v[i]+0.5*dt*k2)/a1
        if Mach1>=Mach[0] and Mach1<=Mach[M1-1]:
            CD1=interpolation(Mach_and_CD,Mach1) 
        else:
            CD1=interp(Mach1,Mach,CD) 
        P3=v[i]+0.5*dt*k2
        k3=-g-(0.5*A*((v[i]+0.5*dt*k2)*(v[i]+0.5*dt*k2))*rho1*CD1/(M[i]+0.5*dt*L2))-(V/(M[i]+0.5*dt*L2))*DMT
        L3=-DMT
        if y[i]+dt*P3>=Altitude[0] and y[i]+dt*P3<=Altitude[M2-1]:
            rho1=interpolation(Altitude_and_Density,y[i]+dt*P3)
            a1=interpolation(Altitude_and_Sound_speed,y[i]+dt*P3)
        else:
            rho1=interp(y[i]+dt*P3,Altitude,Density) 
            a1=interp(y[i]+dt*P3,Altitude,Sound_speed)
        Mach1=(v[i]+dt*k3)/a1
        if Mach1>=Mach[0] and Mach1<=Mach[M1-1]:
            CD1=interpolation(Mach_and_CD,Mach1) 
        else:
            CD1=interp(Mach1,Mach,CD)  
        P4=v[i]+dt*k3
        k4=-g-(0.5*A*((v[i]+dt*k3)*(v[i]+dt*k3))*rho1*CD1/(M[i]+dt*L3))-(V/(M[i]+dt*L3))*DMT
        L4=-DMT
        v[i+1]=(v[i]+(dt/6.0)*(k1+2.0*k2+2.0*k3+k4))
        M[i+1]=(M[i]+(dt/6.0)*(L1+2.0*L2+2.0*L3+L4))
        y[i+1]=(y[i]+(dt/6.0)*(P1+2.0*P2+2.0*P3+P4))
    else:
        if y[i]>=Altitude[0] and y[i]<=Altitude[M2-1]:
            rho1=interpolation(Altitude_and_Density,y[i]) 
            a1=interpolation(Altitude_and_Sound_speed,y[i])
        else:    
            rho1=interp(y[i],Altitude,Density) 
            a1=interp(y[i],Altitude,Sound_speed) 

        Mach1=v[i]/a1
        if Mach1>=Mach[0] and Mach1<=Mach[M1-1]:
            CD1=interpolation(Mach_and_CD,Mach1) 
        else:
            CD1=interp(Mach1,Mach,CD)
        P1=v[i]
        k1=-g-(0.5*A*(v[i]*v[i])*rho1*CD1/M[i])
        if y[i]+0.5*dt*P1>=Altitude[0] and y[i]+0.5*dt*P1<=Altitude[M2-1]:
            rho1=interpolation(Altitude_and_Density,y[i]+0.5*dt*P1)
            a1=interpolation(Altitude_and_Sound_speed,y[i]+0.5*dt*P1)
        else:
            rho1=interp(y[i]+0.5*dt*P1,Altitude,Density) 
            a1=interp(y[i]+0.5*dt*P1,Altitude,Sound_speed) 
        Mach1=(v[i]+0.5*dt*k1)/a1
        if Mach1>=Mach[0] and Mach1<=Mach[M1-1]:
            CD1=interpolation(Mach_and_CD,Mach1) 
        else:
            CD1=interp(Mach1,Mach,CD) 
        P2=v[i]+0.5*dt*k1
        k2=-g-(0.5*A*((v[i]+0.5*dt*k1)*(v[i]+0.5*dt*k1))*rho1*CD1/(M[i]))
        if y[i]+0.5*dt*P2>=Altitude[0] and y[i]+0.5*dt*P2<=Altitude[M2-1]:
            rho1=interpolation(Altitude_and_Density,y[i]+0.5*dt*P2)
            a1=interpolation(Altitude_and_Sound_speed,y[i]+0.5*dt*P2)
        else:
            rho1=interp(y[i]+0.5*dt*P2,Altitude,Density)
            a1=interp(y[i]+0.5*dt*P2,Altitude,Sound_speed) 
        Mach1=(v[i]+0.5*dt*k2)/a1
        if Mach1>=Mach[0] and Mach1<=Mach[M1-1]:
            CD1=interpolation(Mach_and_CD,Mach1) 
        else:
            CD1=interp(Mach1,Mach,CD)
        P3=v[i]+0.5*dt*k2
        k3=-g-(0.5*A*((v[i]+0.5*dt*k2)*(v[i]+0.5*dt*k2))*rho1*CD1/(M[i]))
        if y[i]+dt*P3>=Altitude[0] and y[i]+dt*P3<=Altitude[M2-1]:
            rho1=interpolation(Altitude_and_Density,y[i]+dt*P3)
            a1=interpolation(Altitude_and_Sound_speed,y[i]+dt*P3)
        else:
            rho1=interp(y[i]+dt*P3,Altitude,Density) 
            a1=interp(y[i]+dt*P3,Altitude,Sound_speed)
        Mach1=(v[i]+dt*k3)/a1
        if Mach1>=Mach[0] and Mach1<=Mach[M1-1]:
            CD1=interpolation(Mach_and_CD,Mach1) 
        else:
            CD1=interp(Mach1,Mach,CD)
        P4=v[i]+dt*k3
        k4=-g-(0.5*A*((v[i]+dt*k3)*(v[i]+dt*k3))*rho1*CD1/(M[i]))
        v[i+1]=v[i]+(dt/6.0)*(k1+2.0*k2+2.0*k3+k4)
        M[i+1]=1000.0/g
        y[i+1]=y[i]+(dt/6.0)*(P1+2.0*P2+2.0*P3+P4)
        
    t[i+1]=t[i]+dt  
    i=i+1
    print('---------i---------')
    print(i)
    print('---------t---------')
    print(t[i])   #time of launch
    print('---------M---------')
    print(M[i])    #mass of the aerospace vehicle
    print('---------v---------')
    print(v[i])     # velocity of aerospace vehicle
    print('---------y---------')
    print(y[i])       #Altitude of aerospace vehicle 
        
fig=matplotlib.pyplot.figure(figsize=(10.0,3.0))
axes1 = fig.add_subplot(1,3,1)
axes1.set_ylabel('Mass')
axes1.set_xlabel('Time')
matplotlib.pyplot.plot(t,M,'r^')  #ploting the data points obtained with the numerical modeling 
axes2 = fig.add_subplot(1,3,2)
axes2.set_ylabel('Velocity')
axes2.set_xlabel('Time')
matplotlib.pyplot.plot(t,v,'g^')  
axes3 = fig.add_subplot(1,3,3)
axes3.set_ylabel('Altitude')
axes3.set_xlabel('Time')
matplotlib.pyplot.plot(t,y,'b^')  
matplotlib.pyplot.show()


