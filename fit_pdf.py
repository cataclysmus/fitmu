#!/usr/bin/python2.7

import getopt, sys, os, shutil
from subprocess import call, Popen
from string import replace
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize.minpack import curve_fit
from scipy.optimize import fmin
from scipy.stats import norm, t
from scipy.integrate import quad, cumtrapz
from scipy.stats.mstats import moment
import math


cutoff=6.0
cutoff2=cutoff*cutoff
rdfstep=0.001
atomtype=1
distance=np.arange(0.000001,cutoff+rdfstep,rdfstep)
Nbins=int(cutoff/rdfstep)+1

samples=20
#samplelist="nanocryst/Ge_list.dat"
Ntoms=9568

opts, args = getopt.getopt(sys.argv[1:], "s:a:", ["dump=","exafs","fft","ref="])
for o, a in opts:
    if o == "--dump":
         dumpfile=a
    elif o == "-s":
         rdfstep=float(a)
    elif o == "-a":
         atomtype=a
    else:
         assert False, "unhandled option"


def mkdir_p(path):
   if not os.path.exists(path):
      os.mkdir(path)

#bins=np.loadtxt("ge_4nm_pair.dat", usecols=(1,2))
#bins=np.loadtxt("rdf.dat.3.3_9nm")
bins=np.loadtxt("rdf.dat.1.1")

print bins.shape

#First shell
firstshell_ind=np.where((bins[:,0] >= 2.0) & (bins[:,0] <= 3.1))
firstshell=bins[firstshell_ind]

xfit=firstshell[:,0]
yfit=firstshell[:,1]

xdif=np.diff(xfit)
NN_1=(yfit[1:]*xdif*4*np.pi*xfit[1:]*xfit[1:]).sum()

NN_raw = cumtrapz(4*np.pi*yfit*xfit*xfit, xfit)[-1]

#print NN_1, cumtrapz(4*np.pi*yfit*xfit*xfit, xfit)[-1]

#xdif=np.diff(bins[firstshell_ind])
#xdif=np.append(xdif,xdif[0,0])
#vol=np.power(bins[0,firstshell_ind],3)*4/3*np.pi
#firstshell=(bins[1,firstshell_ind]*xdif).sum()


#Second shell
#secondshell_ind=np.where((bins[0,:]>3.5) & (bins[0,:]<4.3))
#xdif=np.diff(bins[0,secondshell_ind])
#xdif=np.append(xdif,xdif[0,0])
#secondshell=(bins[1,secondshell_ind]*xdif).sum()

#Third shell
#thirdshell_ind=np.where((bins[0,:]>4.3) & (bins[0,:]<5.1))
#xdif=np.diff(bins[0,thirdshell_ind])
#xdif=np.append(xdif,xdif[0,0])
#thirdshell=(bins[1,thirdshell_ind]*xdif).sum()




#norm_fact=firstshell/rdfstep
norm_fact=4/rdfstep

gaus= lambda x, A, mu, ss: A*np.exp(-(x-mu)*(x-mu)/ss/ss/2)/np.sqrt(2.0*np.pi*ss*ss)

#Gram-Charlier A series
#gaus_GC= lambda x, A, r, sig, C3: A*np.exp(-1/2*np.power((x-r)/sig,2))/np.sqrt(2.0*np.pi*sig*sig)*(1+C3/6/np.power(sig,3)*(np.power((x-r)/sig,3) - 3 * (x-r)/sig))

def inf_integrate_2(x, norm, mu, ss):
    func = lambda t: np.exp( -(ss*t)*(ss*t)/2 )*np.cos((x-mu)*t)
    y, err = quad(func, -np.inf, np.inf, epsabs=1e-10, limlst=100)
    return norm*y/np.pi/2

def inf_integrate_3(x, A, mu, ss, C3):
    func = lambda t: np.exp(-(ss*t)*(ss*t)/2 )*np.cos(x*t -mu*t+C3/6*t*t*t)
    y, err = quad(func, -np.inf, np.inf, epsabs=1e-10, limlst=100)
    return A*y/np.pi/2

def inf_integrate_4(x, A, mu, ss, C3, C4):
    func = lambda t: np.exp(-(ss*t)*(ss*t)/2 + C4*t*t*t*t/24 )*np.cos(x*t -mu*t+C3/6*t*t*t)
    y, err = quad(func, -np.inf, np.inf)
    return A*y/np.pi/2




vfunc_2 = np.vectorize(inf_integrate_2)
vfunc_3 = np.vectorize(inf_integrate_3)
vfunc_4 = np.vectorize(inf_integrate_4)



#gaus_skew= lambda x, A1, r1, sig1, alpha1, A2, r2, sig2, alpha2: A1/sig1*norm.pdf((x-r1)/sig1)*norm.cdf(alpha1*(x-r1)/sig1)+A2/sig2*norm.pdf((x-r2)/sig2)*norm.cdf(alpha2*(x-r2)/sig2)
#gaus_skew= lambda x, A1, r1, sig1, alpha1: A1/sig1*norm.pdf((x-r1)/sig1)*norm.cdf(alpha1*(x-r1)/sig1)



#params,cov = curve_fit(gaus_skew, xfit, yfit, p0=[0.0025, 2.45,0.2, 0.0])

#params,cov = curve_fit(vfunc_3, xfit, yfit, p0=[5.28444950e-02, 2.444,0.01, 0.0])
#norm_int, r_int, sig_int, C3_int = params
#print params

#def inf_integrate_3_global(x):
#    global norm_int, r_int, sig_int, C3_int
#    func = lambda t: np.exp(-(sig_int*t)*(sig_int*t)/2 )*np.cos(x*t -r_int*t+C3_int/6*t*t*t)
#    y, err = quad(func, -np.inf, np.inf, epsabs=1e-10, limlst=100)
#    return norm_int*y*2*x*x
#vfunc_g = np.vectorize(inf_integrate_3_global)

#print "\"Cumulant exp\" ", quad(vfunc_g, xfit.min(), xfit.max())[0], r_int, sig_int*sig_int, C3_int

#params,cov = curve_fit(vfunc_4, xfit, yfit, p0=[5.28444950e-02, 2.44396008e+00,1.45415177e-02, -2.37613195e-08, 0.0])


#A, r, sig, C3 = params
#perr = np.sqrt(np.diag(cov))
#print params, perr

m0=cumtrapz(yfit, xfit)[-1]
m1=cumtrapz(yfit*xfit, xfit)[-1]/m0
m2=cumtrapz(yfit*xfit*xfit, xfit)[-1]/m0
m3=cumtrapz(yfit*xfit*xfit*xfit, xfit)[-1]/m0
m4=cumtrapz(yfit*xfit*xfit*xfit*xfit, xfit)[-1]/m0

C1=m1;
C2=m2-m1*m1;
C3=2.0*m1*m1*m1-3.0*m1*m2+m3
C4=-6.0*m1*m1*m1*m1+12*m1*m1*m2-3*m2*m2-4*m1*m3+m4
print "\"Raw cumulants\" ", NN_raw, C1, C2, C3 


#Fit Normal
params,cov = curve_fit(gaus, xfit, yfit, p0=[1.0, 2.45,0.03])
print params
norm_gaus, r_gaus, sig_gaus = params
perr = np.sqrt(np.diag(cov))
gaus_global_rcf= lambda x: norm_gaus*np.exp(-(x-r_gaus)*(x-r_gaus)/sig_gaus/sig_gaus/2)/np.sqrt(2.0*np.pi*sig_gaus*sig_gaus)*4*np.pi*x*x
print "\"Gaus fit\" ", quad(gaus_global_rcf, xfit.min(), xfit.max())[0],  r_gaus, sig_gaus*sig_gaus, 0.0 #, perr


#Fit Normal Skewness
pdf= lambda x, normf, mu, sig, skew: norm.pdf((x-mu)/sig)*norm.cdf(skew*(x-mu)/sig)/sig*normf
params,cov = curve_fit(pdf, xfit, yfit, p0=[4.06422193e-02, 2.45,0.02, 5.0])
perr = np.sqrt(np.diag(cov))
norm_skew, r_skew, sig_skew, alpha=params
delta=alpha/np.sqrt(alpha*alpha+1.0)
b=np.sqrt(2.0/np.pi)
muz=b*delta
sigz=1.0-muz*muz
C1z=r_skew+muz*sig_skew
C2z=(sig_skew*sig_skew*sigz)
C3z=(4.0-np.pi)*(sig_skew*muz)*(sig_skew*muz)*(sig_skew*muz)/2.0
gaus_skew_global_rcf= lambda x: norm.pdf((x-r_skew)/sig_skew)*norm.cdf(alpha*(x-r_skew)/sig_skew)/sig_skew*norm_skew*4*np.pi*x*x
print "\"NS fit\" ",  quad(gaus_skew_global_rcf, xfit.min(), xfit.max())[0], C1z, C2z, C3z #, perr

#Fit Student Skewness
def student(x, normf, mu, sig, skew, nu):
    return map ( lambda y: 2.*t.pdf((y-mu)/sig, nu)*t.cdf(skew*((y-mu)/sig)*np.sqrt((nu+1.0)/(nu+(y-mu)*(y-mu)/sig/sig)),(nu+1.0))/sig*normf if ( (y <3.05) & (y>2.0) ) else 0., x)

def studentv(x, normf, mu, sig, skew, nu):
    if ( (x <3.05) & (x>2.0) ):
        return 2.*t.pdf((x-mu)/sig, nu)*t.cdf(skew*((x-mu)/sig)*np.sqrt((nu+1.0)/(nu+(x-mu)*(x-mu)/sig/sig)),(nu+1.0))/sig*normf
    else:
        return 0.

params=[1.0, 2.44, 0.01, 1.0, 100.0]
params,cov = curve_fit(student, xfit, yfit, p0=params)
perr = np.sqrt(np.diag(cov))
norm_st, r_st, sig_st, alpha_st, freed_d=params
print "sdada", norm_st, r_st, sig_st, alpha_st, freed_d
#print norm_st, r_st, sig_st, alpha_st, freed_d

#def student_skew_global_rcf(x):
#    global norm_st, r_st, sig_st, alpha_st, freed_d
#    return student(x, norm_st, r_st, sig_st, alpha_st, freed_d)*x*x
#= lambda x:  student(x, norm_st, r_st, sig_st, alpha_st, freed_d)*4*np.pi*x*x #2*t.pdf((x-r_st)/sig_st, freed_d)*t.cdf(alpha_st*((x-r_st)/sig_st)*np.sqrt((freed_d+1)/(freed_d+(x-r_st)*(x-r_st)/sig_st/sig_st)),(freed_d+1))/sig_st*norm_st*4*np.pi*x*x
student_skew_global= lambda x: student(x, norm_st, r_st, sig_st, alpha_st, freed_d) #2*t.pdf((x-r_st)/sig_st, freed_d)*t.cdf(alpha_st*((x-r_st)/sig_st)*np.sqrt((freed_d+1)/(freed_d+(x-r_st)*(x-r_st)/sig_st/sig_st)),(freed_d+1))/sig_st*norm_st
m0=cumtrapz(student_skew_global(xfit), xfit)[-1]
m1=cumtrapz(student_skew_global(xfit)*xfit, xfit)[-1]/m0
m2=cumtrapz(student_skew_global(xfit)*xfit*xfit, xfit)[-1]/m0
m3=cumtrapz(student_skew_global(xfit)*xfit*xfit*xfit, xfit)[-1]/m0
m4=cumtrapz(student_skew_global(xfit)*xfit*xfit*xfit*xfit, xfit)[-1]/m0

C1_st=m1;
C2_st=m2-m1*m1;
C3_st=2.0*m1*m1*m1-3.0*m1*m2+m3
C4_st=-6.0*m1*m1*m1*m1+12*m1*m1*m2-3*m2*m2-4*m1*m3+m4
print "\"ST fit\" ",  cumtrapz(student_skew_global(xfit)*xfit*xfit*4*np.pi, xfit)[-1], C1_st, C2_st, C3_st, C4_st #, perr
studentvv=np.vectorize(studentv)
print "Norm", quad(studentvv, 2.2, 2.8, args=(1.0, 2.44687948, 0.02864846, -0.01962208,  0.84210856))






#mean=r
#mode=fmin( vfunc_g, 2.45 )
#DW=sig*sig
#print "Mean=", str(mean), " Mode=", str(mode), " DW=", str(DW), " Skew=", str(C3)


#var=params[2]*params[2] #*(1-2*delta*delta/np.pi)
#skew=(4-np.pi)/2*np.power((delta*np.sqrt(2/np.pi)),3)/np.power((1-2*delta/np.pi),1.5)

#pdf= lambda x: params[0]/params[2]*norm.pdf((x-params[1])/params[2])*norm.cdf(params[3]*(x-params[1])/params[2])

#m0=quad(pdf, 0.0, 6.0)
#pdf1= lambda x: x*params[0]/params[2]/m0[0]*norm.pdf((x-params[1])/params[2])*norm.cdf(params[3]*(x-params[1])/params[2])
#m1=quad(pdf1, 0.0, 6.0)
#pdf2= lambda x: (x-m1[0])*(x-m1[0])*params[0]/params[2]/m0[0]*norm.pdf((x-params[1])/params[2])*norm.cdf(params[3]*(x-params[1])/params[2])
#pdf3= lambda x: (x-m1[0])*(x-m1[0])*(x-m1[0])*params[0]/params[2]/m0[0]*norm.pdf((x-params[1])/params[2])*norm.cdf(params[3]*(x-params[1])/params[2])

#m2=quad(pdf2, 0.0, 6.0)
#m3=quad(pdf3, 0.0, 6.0)

#print firstshell, secondshell, thirdshell, params
#print "Mean=", str(mean), " Var=", str(var), " Skew=", str(skew)
#print "m1=", str(m1[0]), " m2=", str(m2[0]), " m3=", str(m3[0]), "m0=", str(m0[0])


plt.plot(xfit, yfit)
#plt.plot(bins[0], gaus_skew(bins[0], params[0], params[1],  params[2],params[3], params[4], params[5],  params[6],params[7]  ), 'o',label="fit")

#plt.plot(xfit, gaus(  xfit, A, r, sig), 'o',label="fit")
#plt.plot(xfit, vfunc_3( xfit, A, r, sig, C3), 'o',label="int")

#plt.plot(xfit, gaus(  xfit, norm_gaus, r_gaus, sig_gaus), 'o',label="fit")
#plt.plot(xfit, pdf(  xfit, norm_skew, r_skew, sig_skew, alpha), 'o',label="SKEW")
#plt.plot(xfit, vfunc_3( xfit, norm_int, r_int, sig_int, C3_int), 'o',label="int")


#plt.plot(xfit, student(  xfit, norm_st, r_st, sig_st, alpha_st, freed_d), 'o',label="dada")
plt.plot(xfit, student(xfit, 0.84888383, 2.45681867 ,  0.02776907 ,-21.28920008 ,  9.44741797)/xfit/xfit/4./np.pi, 'o',label="SKEW 1")
plt.plot(xfit, student(xfit, 0.84888383, 2.45691 ,  0.02788 ,-6170.66657 ,  9.58376)/xfit/xfit/4./np.pi, 'o',label="SKEW 2")
plt.plot(xfit, student(xfit, 0.84888383,  2.48183 ,0.08326,  -6409.47490 ,  3.15195 )/xfit/xfit/4./np.pi, 'o',label="SKEW 3")




#plt.plot(xfit, vfunc_2( xfit, A, C1, C2), 'o',label="int")





plt.legend()
plt.xlim(2.0,3.0)
plt.show()

#plt.plot(kspace, np.log(four).real, 'b-', kspace, np.log(four).imag, 'r--')
#plt.plot(kspace, (maclaurin(kspace,1,C1)+maclaurin(kspace,2,C2)+maclaurin(kspace,3,C3)).real, 'g-')
#plt.show()