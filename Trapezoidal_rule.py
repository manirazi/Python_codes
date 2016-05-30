import sys
import numpy
import matplotlib.pyplot
def Func(x):
    return pow(x,2)

def Trapz():
    A=float(input("Please enter the location of the 1D domain left bound A: "))
    B=float(input("Please enter the location of the 1D domain right bound B: "))
    M=int(input("Please enter your desired number of subintervals M: "))
    H=float(abs((B-A)/M))  #computing the spatial increment
    SUM=0.0
    for K in range(1,M):   #loop over subintervals
        X=A+float(K)*H
        SUM=SUM+Func(X)
       
        
    SUM=H*(Func(A)+Func(B)+2.0*SUM)/2.0  #computing the trapezoidal sum
    print('The approximate value of the integral of your desired function on the interval',A,'==>',B,'using',M,'subintervals is',SUM)
    

Trapz()




