import sys   #loading libraries
import numpy
import matplotlib.pyplot
import pandas as pd
import operator
import math
from scipy import interp

def interpolation(data,point,check):
    M=len(data.x)     #number of grid points M  
    if check!=1 and check!=2:
        print('Usage note: please set check only to 1 or 2')
        return
    else:
        if check==2:
            col_list=list(data)
            col_list[0],col_list[1]=col_list[1],col_list[0]
            data.columns=col_list
  
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
            return Interpolated_value
            break

def resolving_imputed_variables(filename):
    data = numpy.loadtxt(filename)
    number_of_columns=(data.shape)
    if number_of_columns[1] != 2: # not a proper data set dimension, so print help message
        print('Usage: number of the culmns for the filenames\n \
               must be only 2')
        return
    
    data=pd.DataFrame(data,columns=['x','f'])  #assigning column names to data
    print('---------------original data set-------------------')
    print(data)
    print('---------------------------------------------------')
    index=data['f'].index[data['f'].apply(numpy.isnan)]
    data_index=data.index.values.tolist()
    nan_index_f=[data_index.index(i) for i in index]
    index_f=index
    
    index=data['x'].index[data['x'].apply(numpy.isnan)]
    data_index=data.index.values.tolist()
    nan_index_x=[data_index.index(i) for i in index]
    index_x=index
    data_temp=data                                                
    data_temp=data_temp.dropna(axis=0,how='any')   #removing rows with missing values
    M=len(data_temp.x)     #number of grid points M in temp data set
      #sorting the data on x
    data_temp=data_temp.sort_values(by=['x'])
    
    
    if len(nan_index_f)>=len(nan_index_x):
        j=[-100 for i in range(len(nan_index_f))]
    else:
        j=[-100 for i in range(len(nan_index_x))]
    l=1
    for i in index_f:
        for k in index_x:  #finding rows and columns both with NaN
            if i==k:
                j[l]=k
                l=l+1

            
    
    for i in index_f:
        double_nan=0
        for l in range(len(j)):  #not considering rows and columns both with NaN
            if i==j[l]:
                double_nan=1
                break
        if double_nan==0:
            if data.x[i]>=data_temp.x.iloc[0] and data.x[i]<=data_temp.x.iloc[M-1]:
                data.f[i]=interpolation(data_temp,data.x[i],1)
            else:
                data.f[i]=interp(data.x[i],data_temp.x,data_temp.f)


            

    data_temp=data_temp.sort_values(by=['f'])  #sorting the data on x
    for i in index_x:
        double_nan=0
        for l in range(len(j)):   #not considering rows and columns both with NaN
              if i==j[l]:
                    double_nan=1
                    break
        if double_nan==0:
            if data.f[i]>=data_temp.f.iloc[0] and data.f[i]<=data_temp.f.iloc[M-1]:
                data.x[i]=interpolation(data_temp,data.f[i],2)
            else:
                data.x[i]=interp(data.f[i],data_temp.f,data_temp.x)
                
    data=data.dropna(axis=0,how='any')   #removing rows with two missing values
    data=data.sort_values(by=['x','f'],ascending=[1,0])   #sorting the data

    print('---------------Modified data set-------------------')
    print(data)
    print('---------------------------------------------------')

                        
                        
                    
script = sys.argv[0]
filenames = sys.argv[1:]
if len(filenames) == 0: 
    print('You must enter a data file name, please try again')
else:
    K=0
    for f in filenames:
        K=K+1
        print("For data file :",f)
        print('===========modified data sets with no imputed variable=============')
        print(K,'filename: ',f)
        resolving_imputed_variables(f)
        print('========================')  
            
    