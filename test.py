#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 17:54:50 2019

@author: yael
"""

import math,numpy,random
import datetime
from mpi4py import MPI
print(datetime.datetime.now())
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
runs = 10               # no. of simulations
p = 0.25        # (table 2, HA05)
M_sun = 1.98847542e30   # solar mass [kg]
G = 6.67408e-11         # gravitational constant [m3/(kg s2)]
c = 299792458           # speed of light in vacuum [m/(s)]
pc = 3.08567758e16      # Parsec [m]
Mbh = 3*M_sun*10**6    # mass of central BH (table 2, HA05)
Mco = 10*M_sun            # mass of compact object (table 2, HA05)
rs = 2*G*Mbh/(c**2)    # Schwarzschild radius of MBH
rh = 2*pc        # radius of influence (table 2, HA05)
Nh = 2*Mbh/M_sun        # number of stars within rh (section 5, HA05)
t_h = 3.1415e16    # relaxation time at rh (table 2, HA05)
fs = 0.001              # number fraction (table 2, HA05)
J_lc = 2*math.sqrt(3)*G*Mbh/c    # loss cone momentum (E~0)
s_crit = 0.001          # star is surely in the "dissipation phase"
ac = 0.01*pc            # critical semimajor axis (table 2, HA05)
dc = (ac*(rh**(3/(3-2*p)-1)))**((3-2*p)/3)    # eq. 30, HA05
a4 = numpy.linspace(0.00075, 0.0009, num=2)*pc
a5 = numpy.linspace(0.001, 0.01, num=10)*pc
a6 = numpy.linspace(0.0125, 0.05, num=8)*pc
a7 = numpy.linspace(0.06, 0.1, num=5)*pc
a1 = numpy.append(a4, a5)               
a2 = numpy.append(a6, a7)
a0 = numpy.append(a1, a2)               # initial semimajor axis, range: .1-.001pc (fig. 3, HA05)
def x():                                # +-1 with equal probabilities
    return 1 if random.random() < 0.5 else -1
DFJa0J = []
ecc = []
a_final = []
PRECISION = 0.03
nsa0 = 0
nproblem = 0
for i in range(runs):
    a = a0[rank+20]
    #rp = (a*(2*Mco/Mbh)**(1.0/3.0))*(random.random()+0.5)
    #e = 1-rp/a
    Jmax = math.sqrt((numpy.absolute((2*a/rs)**2/(2*a/rs-3)))*((G*Mbh/c)**2))
    e = math.sqrt(1-(15*J_lc/Jmax)**2)
    #e = 0.99
    rp = a*(1-e)                                        # initial periapse
    q = 2*(1-e**2)*a/rs
    E = -math.sqrt(numpy.absolute((q-2-2*e)*(q-2+2*e)/(q*(q-3-e**2))))*(c**2)
    J = math.sqrt((numpy.absolute(q**2/(q-3-e**2)))*((G*Mbh/c)**2))
    #s = ((a/dc)**(3/2))*((a/rh)**(-p))*((J/J_lc)**5)        # initial scattering to dissipation ratio
    P = (2*math.pi*a**1.5)/math.sqrt(G*Mbh)
    DiffusionFlag = 1
    while P > 10000 and J >= J_lc:
    #while s > s_crit and J >= J_lc:
        P = (2*math.pi*a**1.5)/math.sqrt(G*Mbh)    # orbital period P(E)
        Jmax = math.sqrt((numpy.absolute((2*a/rs)**2/(2*a/rs-3)))*((G*Mbh/c)**2))            # maximal (circular orbit) angular momentum per energy
        t_r = t_h*((a/rh)**p)                        # relaxation time
        delta1Jsc = P*(Jmax**2)/(2*t_r*J)
        delta2Jsc = Jmax*math.sqrt(P/t_r)
        deltaJGW = (16*math.pi/5)*((1+7*(e**2)/8)/((1+e)**2))*(G*Mco/c)*((rs/rp)**2)
        deltaE = (8*math.pi/(5*math.sqrt(2)))*((1+(73/24)*e**2+(37/96)*e**4)/((1+e)**(7/2)))*(Mco*(c**2)/Mbh)*((rs/rp)**(7/2))    # deltaE_GW (HA06)
        dJ = numpy.absolute(PRECISION*J/deltaJGW)
        dE = numpy.absolute(PRECISION*E/deltaE)
        correction_factor = 7.0*e/8.0 + 1.0/8.0;
        t0 = (2*math.pi*math.sqrt(G*Mbh*a)/deltaE)*correction_factor
        dJscat = (PRECISION*J/delta2Jsc)**2
        if (J > 0.9*Jmax and DiffusionFlag == 1):
            if t0 < 14.7481*3.1415e16:
                DiffusionFlag = 0
            else:
                nproblem += 1
                break
        if(DiffusionFlag == 1):
            nstep = max(1,min(dJ,dE,dJscat))
        else:
            nstep = max(1,min(dJ,dE))
        deltaJ = -nstep*delta1Jsc + x()*DiffusionFlag*math.sqrt(nstep)*delta2Jsc - nstep*deltaJGW
        J = J+deltaJ                    # perturbation in momentum
        E = E+nstep*deltaE              # perturbation in energy
        coeff = [c**4-E**2, -2*G*Mbh*c**2, c**2*J**2, -2*G*Mbh*J**2]
        r = numpy.roots(coeff)
        if numpy.iscomplex(r).any():
            if J < 4*G*Mbh/c:          # J_LastStableOrbit = 4*G*Mbh/c
                break
            if J >= Jmax:
                i -= 1
                break
            else:
                nproblem += 1
                break
        rmin = numpy.argwhere(r==numpy.amin(r))
        r = numpy.delete(r, rmin)
        rp = numpy.amin(r)        # periapse, intermediate root of Eq. (36), HA05
        ra = numpy.amax(r)        # apoapse, largest root of Eq. (36), HA05
        a = (rp+ra)/2                   # updated semimajor axis
        e = 1-rp/a                      # updated eccentricity
        if (e<0):
            break
        #if (1-e**2)*a < (6+2*e)*rs:
            #break
        #s = ((a/dc)**(3/2))*((a/rh)**(-p))*((J/J_lc)**5)
    #if J >= Jmax:
        #break
    if (P <= 10000):
    #if (s <= s_crit):
        DFJa0J.append(J/J_lc)
        ecc.append(e)
        a_final.append(a)
        nsa0 += 1
f = open( 'file102.txt', 'a' )
f.write( repr(rank) + '\n' + repr(datetime.datetime.now()) + '\n' + repr(nsa0) + '\n' + repr(nproblem) + '\n' + repr(DFJa0J) + '\n' + repr(ecc) + '\n' + repr(a_final) + '\n' )
f.close()