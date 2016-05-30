import sys   #loading libraries
import numpy
import matplotlib.pyplot
import pandas as pd
import operator
import math


def Trapz(filename):
    #data = numpy.loadtxt('data.dat')  #reading data file
    data = numpy.loadtxt(filename)
    number_of_columns=(data.shape)
    if number_of_columns[1] != 2: # not a proper data set dimension, so print help message
        print('Usage: number of the culmns for the filenames\n \
               must be only 2')
        return
    
    data=pd.DataFrame(data,columns=['x','f'])  #assigning column names to data
    data=data.dropna(axis=0,how='any')   #removing rows with missing values
    print(data)
    A=min(data.x)  #location of the 1D domain left bound A
    B=max(data.x)   #location of the 1D domain right bound B
    M=len(data.x)     #number of grid points M
    #data=data.sort(['x','f'],ascending=[1,0])
    data=data.sort_values(by=['x','f'],ascending=[1,0])   #sorting the data
    
    SUM=0.0
    for K in range(0,M-1):   #loop over subintervals  
        SUM=SUM+0.5*(data.x.iloc[K+1]-data.x.iloc[K])*(data.f.iloc[K+1]+data.f.iloc[K])

        
    print('The approximate value of integral over the provided data set is: ',SUM)
    
def ScatteredDataIntegration(filename):
        #data = numpy.loadtxt('data.dat')  #reading data file
    data = numpy.loadtxt(filename)
    number_of_columns=(data.shape)
    if number_of_columns[1] != 2: # not a proper data set dimension, so print help message
        print('Usage: number of the culmns for the filenames\n \
               must be only 2')
        return
        
    data=pd.DataFrame(data,columns=['x','f'])  #assigning column names to data
    data=data.dropna(axis=0,how='any')   #removing rows with missing values
    print(data)
    A=min(data.x)  #location of the 1D domain left bound A
    B=max(data.x)   #location of the 1D domain right bound B
    M=len(data.x)     #number of grid points M
    #data=data.sort(['x','f'],ascending=[1,0])
    data=data.sort_values(by=['x','f'],ascending=[1,0])   #sorting the data
    ia=0  #Points index starts from 0
    ib=M-1      # ends at M-1
    n=M-1
    if n<3 or ia>ib:
        print('the integration method failed')
        if ia>ib:
            print('Usage: do not use an empty data file')
        if n<3:
            print('Usage: the number of data point for the filenames\n \
                   must be greater than 4')
        return
    else:
        if ia==ib:
            result=0.0
            error=0.0
        else:
            INT=0.0
            error=0.0
            s=0.0
            c=0.0
            r4=0.0
            if ia==n-1 and M>4:
                j=n-2
            else:
                if ia>1:
                    j=ia
                else:
                    j=2
            if ib==1 and M>4:
                k=3
            else:
                if n>ib+2:
                    k=ib+1
                else:
                    k=n-1
        for i in range(j,k+1):
            if i==j:
                h2=data.x.iloc[j-1]-data.x.iloc[j-2]
                d3=(data.f.iloc[j-1]-data.f.iloc[j-2])/h2
                h3=data.x.iloc[j]-data.x.iloc[j-1]
                d1=(data.f.iloc[j]-data.f.iloc[j-1])/h3
                h1=h2+h3
                d2=(d1-d3)/h1
                h4=data.x.iloc[j+1]-data.x.iloc[j]
                r1=(data.f.iloc[j+1]-data.f.iloc[j])/h4
                r2=(r1-d1)/(h4+h3)
                h1=h1+h4
                r3=(r2-d2)/h1
                if ia==0:
                    INT=h2*(data.f.iloc[0]+h2*(0.5*d3-h2*((d2/6.0)-(h2+2.0*h3)*(r3/12.0))))
                    s=-(h2*h2*h2)*(h2*(3.0*h2+5.0*h4)+10.0*h3*h1)/60.0
            if i!=j:
                h4=data.x.iloc[i+1]-data.x.iloc[i]
                r1=(data.f.iloc[i+1]-data.f.iloc[i])/h4
                r4=h4+h3
                r2=(r1-d1)/r4
                r4=r4+h2
                r3=(r2-d2)/r4
                r4=r4+h1
                r4=(r3-d3)/r4
            if i<=ib and i>ia:
                INT=INT+h3*((data.f.iloc[i]+data.f.iloc[i-1])*0.5-h3*h3*(d2+r2+(h2-h4)*r3)/12.0)
                c=(h3*h3*h3)*(2.0*h3*h3+5.0*(h3*(h4+h2)+2.0*h4*h2))/120.0
                error=error+(c+s)*r4
                if i==j:
                    s=s+2.0*c
                else:
                    s=c
            else:
                error=error+r4*s
            if i==k:
                if ib==n:
                    INT=INT+h4*(data.f.iloc[n]-h4*(r1*0.5+h4*(r2/6.0+(2.0*h3+h4)*r3/12.0)))
                    error=error-(h4*h4*h4)*r4*(h4*(3.0*h4+5.0*h2)+10.0*h3*(h2+h3+h4))/60.0
                if ib>=n-1:
                    error=error+s*r4   
            else:
                h1=h2
                h2=h3
                h3=h4
                d1=r1
                d2=r2
                d3=r3
                
                
    result=INT+error
    print('The approximate value of integral over the provided data set is: ',result,error)  
       

              
                    
            
            
        
    

    
script = sys.argv[0]
if len(sys.argv) <2: # no arguments, so print help message
    print('Usage: python Data_Integration.py action filenames\n \
           action must be one of --Trapz --HighAcc\n \
           if filenames is blank, input is taken from stdin;\n \
           otherwise, each filename in the list of arguments is processed in turn')
else:
    action = sys.argv[1]
    assert action in ['--Trapz', '--HighAcc'], \
           'Action is not one of --Trapz or --HighAcc: ' + action
    filenames = sys.argv[2:]
    if len(filenames) == 0: 
        print('You must enter a data file name, please try again')
    else:
        K=0
        if action == '--Trapz':
            for f in filenames:
                K=K+1
                print('===========Trapezoidal Rule=============')
                print(K,'filename: ',f)
                Trapz(f)
                print('========================')        
        elif action == '--HighAcc':
            for f in filenames:
                K=K+1
                print('===========High Accuracy Numerical Integration=============')
                print(K,'filename: ',f)
                ScatteredDataIntegration(f)
                print('========================')
