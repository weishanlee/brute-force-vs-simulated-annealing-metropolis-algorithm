# -*- coding: utf-8 -*-
"""
This algorithm compares the shortest path calculated from either
    1. Brute force method: calculating all the possible path and find out the 
       minimum path, or 
    2. Simulated annealing and the metropolis algorithm(SAMA).
The randonly generated sites are listed in rr.csv
NN is a number that is larger than 3 and smaller than N=25
Version V0_2: 1. Change the program to loops and calculate the possible NNs automatically
              2. Add function definition def cpu_stats() to keep track of RAM usage.
              3. Add the targetScore to SAMA 
"""
from math import sqrt,exp
import numpy as np
from random import random,randrange
#from vpython import *
import itertools
import time
import pandas as pd
import random as rand
N = 25 # number of sites

# Randomly generate x and y coordinates of N locations 
rr = np.random.rand(N,2)
rr = pd.DataFrame(rr)
dfCSV = rr
dfCSV_file = open("./rr.csv",'w',newline='') 
dfCSV.to_csv(dfCSV_file, sep=',', encoding='utf-8',index=False)
dfCSV_file.close()

# Method 1: brute force method

def findOrders(n):
    """
    Descriptions
    ------------
    Reference: https://stackoverflow.com/questions/51531766/python-algorithms-necklace-generation-circular-permutations
               https://stackoverflow.com/questions/960557/how-to-generate-permutations-of-a-list-without-reverse-duplicates-in-python-us
    Find all possible orders for a given positive integers n,
    excluding mirrored or circular repetitions.
    Example: if n = 3, there in only one possible orders:
            [0, 1, 2]
    Parameters
    ----------
    n : positive integers

    Returns
    -------
    x : list
        x[i][:] refers to the site order in the ith permutation
    """
    
    ls = np.arange(0,n).tolist()
    orders = []
    
    for p in itertools.permutations( ls[1:] ):
        if p <= p[::-1]:
            orders += [p]
    
    for end in orders:
        yield [ls[0]] + list(end) 

def mag(xx):
    return sqrt(xx[0]**2+xx[1]**2)

# Function to calculate the total length of the tour
def distance(r):
    s = 0.0
    for i in range(NN):
        s += mag(r[i+1]-r[i])
    s += mag(r[NN]-r[0])
    return s

def writeLog(msg):
    with open('log.txt', 'a+') as the_file:
        print(msg)
        the_file.write(msg)
        
import os, psutil  

def cpu_stats():
    pid = os.getpid()
    py = psutil.Process(pid)
    memory_use = py.memory_info()[0] / 2. ** 30
    return 'Memory: ' + str(np.round(memory_use, 2)) + 'GB\t'

def bruteForce(NN, r):
    x = list(findOrders(NN))
    print(cpu_stats())
    distances = []

    r_copy = np.copy(r)

    start = time.time()

    for i in range( len(x) ):
        for j in range(NN):
            r[j,0] = r_copy[x[i][j],0]
            r[j,1] = r_copy[x[i][j],1]
            r[j,2] = 0.0
            r[j,3] = j
        r[NN,0] = r[0,0]
        r[NN,1] = r[0,1]
        r[NN,2] = r[0,2]
        r[NN,3] = NN    
    
        distances.append( distance(r) ) 
        #print("All possible distances: {}".format(distances))        
    end = time.time()
    writeLog("Number of sites: {}:\t".format(NN))
    #writeLog(cpu_stats())
    writeLog("The smallest value by brute force: {:.5f}\t".format( np.min(distances)))
    writeLog("Time elpased: {:.5f}\t".format(end-start))
    return end-start, np.min(distances)

# Method 2: Simulated annealing and the metropolis algorithm.
R = 0.02
Tmax = 1.0
Tmin = 1e-2
tau = 1e3

def TSP(NN, r, smallestDistance):
    score = distance(r)
    #$initScore = score
    #minScore = initScore
    #print("Initial score = {:.5f}\n".format(initScore))
    #tRecord = []
    #scoreRecord = []

    t0=0 # setting up the beginning of the time "lump"
    #tRecord += [0]
    #scoreRecord += [score]

    firstInitial = True
    
    targetScore = smallestDistance
    
    start = time.time()
    while score>targetScore:
    
        if firstInitial == False: 
            # Set up another initial configuration
            randomList = rand.sample(range(0, NN), NN)
    
            r = np.empty([NN+1,4])
            for j in randomList:
                r[j,0] = rr.iloc[j][0]
                r[j,1] = rr.iloc[j][1]
                r[j,2] = 0.0
                r[j,3] = j
            r[NN,0] = r[0,0]
            r[NN,1] = r[0,1]
            r[NN,2] = r[0,2]
            r[NN,3] = NN
        
        #Calculate the initial distance
        score = distance(r)

        """# Set up the graphics
        canvas(center=vector(0.5,0.5,0.0), background = color.white)
        for i in range(NN):
            if i == 0:
                sphere(pos=vector(r[i,0],r[i,1],0.0),radius=R,color = color.blue)
            else:
                sphere(pos=vector(r[i,0],r[i,1],0.0),radius=R,color = color.black)
            l = curve(pos=r.tolist(),radius=R/2,color = color.red)"""

        # Main loop
        t = 0
        T = Tmax
        
        while T>Tmin:
        
            # Cooling
            t += 1
            T = Tmax*exp(-t/tau)

            """# Update the visualization every 100 moves
            if t%100==0:
                rate(50)
                for i in range(NN+1):
                    pos = vector(r[i,0],r[i,1],0.0)
                    l.modify(i,pos)"""

            # Choose two cities to swap and make sure they are distinct
            i,j = randrange(1,NN),randrange(1,NN)
            while i==j:
                i,j = randrange(1,NN),randrange(1,NN)

            # Swap them and calculate the change in distance
            oldScore = score
            r[i,0],r[j,0] = r[j,0],r[i,0]
            r[i,1],r[j,1] = r[j,1],r[i,1]
            score = distance(r)
            deltaScore = score - oldScore
            
            
            try:
                ans = np.exp(-deltaScore/T)
            except OverflowError:
                if -deltaScore/T > 0:
                    ans = float('inf')
                else:
                    ans = 0.0

            # If the move is rejected, swap them back again
            if random() > ans:
                r[i,0],r[j,0] = r[j,0],r[i,0]
                r[i,1],r[j,1] = r[j,1],r[i,1]
                score = oldScore
                
                if np.abs(score - distance(r))>1e-5:
                    print("score: {}".format(score))
                    print("distance: {}".format(distance()))
                    print("Error Line 205")
                    
            """if score < minScore: 
                minScore = score
                outPutScrVSTime(tRecord, scoreRecord)
                outPutSitesOrder(rCoor)
                dt = datetime.now()
                print(dt.year, '/', dt.month, '/', dt.day, ' ',
                      dt.hour, ':', dt.minute, ':', dt.second)
                print("Delta score = {:.5f}".format(deltaScore))
                print("New score = {:.5f}\n".format(score))"""
                
            """if t%10==0:
                tRecord += [t0+t]
                scoreRecord += [score]"""
        
        t0 = t0 + t # go to next time "lump"
        firstInitial = False
        
    end = time.time()
    writeLog("The smallest value by SAMA: {:.5f}\t".format(score))
    writeLog("Time elpased: {:.5f}\n".format(end-start))
    return end-start, score


NN = 12  # NN is a value smaller than N
# Loop for calculating all NN
nSites = np.arange(3,NN+1).tolist()

timeBruteForce = []
distanceBruteForce = []
    
TimeTSP = []
distanceTSP = []

for i in nSites:
    NN = i     
    r = np.empty([NN+1,4])
    for j in range(NN):
        r[j,0] = rr.iloc[j][0]
        r[j,1] = rr.iloc[j][1]
        r[j,2] = 0.0
        r[j,3] = j
    r[NN,0] = r[0,0]
    r[NN,1] = r[0,1]
    r[NN,2] = r[0,2]
    r[NN,3] = NN

    r_init = np.copy(r)
    
    elapsedTime , smallestDistance = bruteForce(NN, r)
    
    timeBruteForce.append(elapsedTime)
    distanceBruteForce.append(smallestDistance)
    
    r = r_init
    
    elapsedTime , smallestDistance = TSP(NN, r, smallestDistance)
    
    TimeTSP.append(elapsedTime)
    distanceTSP.append(smallestDistance)

# Write to csv
data = {'nSites' : nSites,'smallestTotalDistanceBruteForce' : distanceBruteForce,
                          'elapsedTimeBruteForce' : timeBruteForce,
                          'smallestTotalDistanceTSP' : distanceTSP,
                          'elapsedTimeTSP' : TimeTSP}
dfCSV = pd.DataFrame(data)
dfCSV_file = open("./bruteForceVsTSP.csv",'w',newline='') 
dfCSV.to_csv(dfCSV_file, sep=',', encoding='utf-8',index=False)
dfCSV_file.close()    
    
# plot bruteForceVsTSP.csv
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
plt.figure()
plt.title("Comparisons of Computation Time Between Brute-force and TSP")
ax = plt.gca()
df = pd.read_csv("./bruteForceVsTSP.csv") 
plt.plot(df.nSites,df.elapsedTimeBruteForce,'bo-', df.nSites,df.elapsedTimeTSP,'ro-')
plt.minorticks_on()
minorLocatorX = AutoMinorLocator(2) # number of minor intervals per major # inteval
minorLocatorY = AutoMinorLocator(4)
ax.set_xlabel("Number of Sites",size = 16)
ax.set_ylabel("Computation Time (sec)",size = 16)
ax.xaxis.set_minor_locator(minorLocatorX) # add minor ticks on x axis
ax.yaxis.set_minor_locator(minorLocatorY) # add minor ticks on y axis
plt.xlim(2,13)
plt.ylim(-200,1600)
plt.grid(True)
plt.show()   
plt.savefig('fig.jpg') #, format='eps')