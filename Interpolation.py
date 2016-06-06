import sys   #loading libraries
import numpy
import matplotlib.pyplot
import pandas as pd
import operator
import math

# this code computes the desired value if it is missing or asked by the user using the data in the data file

def interpolation(filename,point):
    data = numpy.loadtxt(filename)
    number_of_columns=(data.shape)
    if number_of_columns[1] != 2: # not a proper data set dimension, so print help message
        print('Usage: number of the culmns for the filenames\n \
               must be only 2')
        return
    
    data=pd.DataFrame(data,columns=['x','f'])  #assigning column names to data
    data=data.dropna(axis=0,how='any')   #removing rows with missing values
    print(data)
    M=len(data.x)     #number of grid points M
    data=data.sort_values(by=['x','f'],ascending=[1,0])   #sorting the data
    
    if point<data.x.iloc[0] or point>data.x.iloc[M-1]: # not a proper data set dimension, so print help message
        print('Your desired x value is out of the range for this data set; please try again')
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
            print('Interpolated value at point x=',point,'is',Interpolated_value)
            fig=matplotlib.pyplot.figure(figsize=(10.0,3.0))
            axes1 = fig.add_subplot(1,3,1)
            axes1.set_ylabel('function')
            axes1.set_xlabel('x')
            matplotlib.pyplot.plot(data.x.iloc[0:M-1],data.f.iloc[0:M-1],'r^')  #ploting the interpolated data points among the data set
            matplotlib.pyplot.plot([point],[Interpolated_value],'go')
            matplotlib.pyplot.show()
            break
                        
                        
                    
script = sys.argv[0]
filenames = sys.argv[1:]
if len(filenames) == 0: 
    print('You must enter a data file name, please try again')
else:
    K=0
    for f in filenames:  #this code can be run on multiple data sets
        K=K+1
        print("For data file :",f)
        point=float(input("Please enter the your desired x value: "))
        print('===========Numerical Interpolation=============')
        print(K,'filename: ',f)
        interpolation(f,point)
        print('========================')  
            
    