from pylab import *
from random import randint
import time
import numpy
from numpy import zeros,exp,cosh,sinh


def pbc(i,limit,add):
    return (i + limit + add) % limit

            
            
# Perform the Metropolis algorithm
def metropolis(nSpins,mcCycles,temp,expectationValues,acceptedFlips,rndm=False):
    #Initialize RNG, can be called by rand(gen) to get a random number between 0 and 1
    #random_device rd;
    #mt19937_64 gen(rd());
    #uniform_real_distribution<double> rand(0.0,1.0);
    if rndm:
        seed(3004)
        spinMatrix = numpy.random.randint(2, size=(nSpins,nSpins))
    else: spinMatrix = zeros(shape=(nSpins,nSpins)) + 1
    
    energy = 0.0
    magneticMoment = 0.0
    tmet_tot = 0.
    for y in range(int(nSpins)):#(int y = 0; y < nSpins; y++)
        for x in range(int(nSpins)):#(int x = 0; x < nSpins; x++)
            magneticMoment += spinMatrix[x,y]
            energy -= spinMatrix[x,y] * (spinMatrix[pbc(x,nSpins,-1),y] + spinMatrix[x,pbc(y,nSpins,-1)])
    energyDifference = zeros(17);
    for dE in range(-8,8+1,4):#(int dE = -8; dE <= 8; dE += 4)
        #print exp(-dE/temp)
        #print dE
        energyDifference[dE + 8] = exp(-dE/temp)
    #print energyDifference
    

    for i in range(int(mcCycles)):
        tmet0 = time.clock()
    #Metropolis
    #Loop over all spins, pick a random spin each time
        for s in range(int(nSpins**2)):
            x = int(numpy.random.random()*nSpins)
            y = int(numpy.random.random()*nSpins)
            deltaE = 2*spinMatrix.item(x,y)*\
                     (spinMatrix.item(pbc(x,nSpins,-1), y) +\
                      spinMatrix.item(pbc(x,nSpins,1),  y) +\
                      spinMatrix.item(x, pbc(y,nSpins,-1)) +\
                      spinMatrix.item(x, pbc(y,nSpins,1)))
            if numpy.random.random() <= energyDifference[int(deltaE)+8]:
                #Accept!
                energy += deltaE
                spinMatrix[x,y] *= -1.0
                magneticMoment += 2 * spinMatrix[x,y]
                acceptedFlips+=1
        
        
        expectationValues[0] += energy
        expectationValues[1] += energy * energy
        expectationValues[2] += abs(energy)
        expectationValues[3] += magneticMoment
        expectationValues[4] += magneticMoment * magneticMoment
        expectationValues[5] += abs(magneticMoment)
    
        
        tmet1 = time.clock()
        tmet_tot += (tmet1 - tmet0)
    acceptedFlips_list.append(acceptedFlips)
    #print 'MetroAddtime: ',tmet_tot
    
    
def calcWrite(nSpins,mcCycles,temp,expectationValues,CvError,XError,acceptedFlips,i):

    # Normalization of the values
    norm = 1./mcCycles
    normSpins = 1./(nSpins*nSpins)
    i += 1
    
    # Numerical values
    expectVal_E = expectationValues[0]*norm#// / nSpins / nSpins;
    expectVal_E2 = expectationValues[1]*norm
    expectVal_Eabs = expectationValues[2]*norm
    expectVal_M = expectationValues[3]*norm#// / nSpins / nSpins;
    expectVal_M2 = expectationValues[4]*norm#// / nSpins / nSpins;
    expectVal_Mabs = expectationValues[5]*norm

    expectVal_Cv = (expectVal_E2 - expectVal_Eabs * expectVal_Eabs) * normSpins / (temp * temp)
    expectVal_X = (expectVal_M2 - expectVal_Mabs * expectVal_Mabs) * normSpins / temp

    #print "\nExpectation values, numerical: "
    #print "Energy: ",expectVal_E
    #print  "Energy^2: ",expectationValues_E2
    #print  "|Energy|: ",expectVal_Eabs
    #print  "Magnetic moment: ",expectVal_M
    #print  "Magnetic moment^2: ",expectationValues_M2
    #print  "|Magnetic moment|: ", expectVal_Mabs
    #print  "Heat capacity: ", expectVal_Cv
    #print  "Susceptibility: ", expectVal_X

    # Analytical values
    J = 1.0;
    beta = 1.0;
    Z = 4*cosh(8*J*beta) + 12;
    expectValAnalytical_E = 32*J*sinh(8*J*beta) / Z;
    expectValAnalytical_M2 = (32*exp(8*J*beta) + 32) / Z;
    expectValAnalytical_Mabs = (8*exp(8*J*beta) + 16) / Z;
    expectValAnalytical_Cv = ((256*J*cosh(8*J*beta)) / Z - expectValAnalytical_E *
                                     expectValAnalytical_E) * normSpins / (temp * temp);
    expectValAnalytical_X = (expectValAnalytical_M2 - expectValAnalytical_Mabs *
                                    expectValAnalytical_Mabs) * normSpins / temp;

    #//cout << "\nExpectation values, analytical:" << endl;
    #//cout << "Energy: " << expectValAnalytical_E << endl;
    #/cout << "Heat capacity: " << expectValAnalytical_Cv << endl;
    #//cout << "Susceptibility: " << expectValAnalytical_X << endl;

    #// Error between numerical and analytical
    #//vec CvError(10);
    #//vec XError(10);
    CvError += abs(expectValAnalytical_Cv - expectVal_Cv) / expectValAnalytical_Cv;
    XError += abs(expectValAnalytical_X - expectVal_X) / expectValAnalytical_X;
    #print CvError/numberOfLoops
    #//cout << "Cv error: " << CvError << endl;
    #//cout << "X error: " << XError << endl;
    
    mcCycles_list.append(mcCycles)
    temp_list.append(temp)
    expectVal_E_list.append(expectVal_E * normSpins)
    expectVal_Cv_list.append(expectVal_Cv)
    expectVal_M_list.append(expectVal_M * normSpins)
    expectVal_X_list.append(expectVal_X)
    expectVal_Mabs_list.append(expectVal_Mabs * normSpins)
    
    CvError_list.append(CvError/i)
    XError_list.append(XError/i)
    



#Lattice size for phase transition    
#nSpins = [20,40,60]
#temperature start for phase transition and accepted flips
tempInit = 2.16
#temperature end for phase transition and accepted flips
tempFinal = 2.34
#temperature step for phase transition
#tempStep = 0.02
#number of MC cycles for phase transition
#numberOfLoops[1E4]
numberOfLoops = [1E5,2E5,3E5,4E5,5E5,6E5,7E5,8E5,9E5,1E6]
#Lattice size for MC cylce equalibrium
nSpins = 2
#temperature start for MC cycles equalibrium
tempInit = 2.4
#temperature end for MC cycles equalibrium
tempFinal = 2.4
#Lattice size for accepted flips
#nSpins = [2]
#temperature step for accepted flips
tempStep = 0.01
#number of cycles for MC cycle equalibrium 2x2
#numberOfLoops = [1E3,2E3,3E3,4E3,5E3,6E3,7E3,8E3,9E3,1E4,2E4,3E4,4E4,5E4,6E4,7E4,8E4,9E4,1E5,2E5,3E5,4E5,5E5,6E5,7E5,8E5,9E5,1E6,2E6,3E6,4E6,5E6]
#number of cucles for MC cycle equalibrium 20x20
#numberOfLoops = [1E3,2E3,3E3,4E3,5E3,6E3,7E3,8E3,9E3,1E4,2E4,3E4,4E4,5E4,6E4,7E4,8E4,9E4,1E5,2E5,3E5]


CvError_list = []
XError_list = []
mcCycles_list = []
temp_list = []
expectVal_E_list = []
expectVal_Cv_list = []
expectVal_M_list = []
expectVal_X_list = []
expectVal_Mabs_list = []
acceptedFlips_list = []
energy = []

CvError = 0.
XError = 0.




acceptedFlips = 0
#print int((tempFinal-tempInit)/tempStep)
if len(numberOfLoops) == 1:
    mcCycles = int(numberOfLoops[0])
    # Start Metropolis algorithm with Monte Carlo sampling
    for nSpins in nSpins:
        CvError_list = []
        XError_list = []
        mcCycles_list = []
        temp_list = []
        expectVal_E_list = []
        expectVal_Cv_list = []
        expectVal_M_list = []
        expectVal_X_list = []
        expectVal_Mabs_list = []
        acceptedFlips_list = []
        energy = []
        for temp in range(int(tempInit/tempStep),int(tempFinal/tempStep)+1,1):
            temp = temp*tempStep
            print 'temp:',temp
            
            #print 'Temperature:', temp
            expectationValues = zeros(6);
            t0 = time.clock()
            metropolis(nSpins, mcCycles, temp, expectationValues, acceptedFlips)
            calcWrite(nSpins, mcCycles, temp, expectationValues, CvError, XError, acceptedFlips,mcCycles);
            t1 = time.clock()
            tim = t1-t0
            print 'Time:',tim,'s'
        figure(2)
        plot(temp_list,expectVal_E_list)
        grid(True)
        xlabel(r'$Temperature$')
        ylabel(r'${<E>}/{L^2}$')
        legend(['L=20','L=40','L=60','L=80','L=100',],)
        figure(3)
        plot(temp_list,expectVal_Mabs_list)
        grid(True)
        xlabel(r'$Temperature$')
        ylabel(r'${<|M|>}/{L^2}$')
        legend(['L=20','L=40','L=60','L=80','L=100',],)
        figure(4)
        plot(temp_list,expectVal_Cv_list)
        grid(True)
        xlabel(r'$Temperature$')
        ylabel(r'${C_V}/{L^2}$')
        legend(['L=20','L=40','L=60','L=80','L=100',],)
        figure(5)
        plot(temp_list,expectVal_X_list)
        grid(True)
        xlabel(r'$Temperature$')
        ylabel(r'${\chi}/{L^2}$')
        legend(['L=20','L=40','L=60','L=80','L=100',],)

        
        
        
    figure(6)
    plot(temp_list,acceptedFlips_list)
    xlabel(r'$Temperature$')
    ylabel(r'$\#\ of\ accepted\ flips$')
    grid(True)
else:
    print '''# of Monte Carlo cycles  |    Cv Error    |    Chi Error    |     Time (s)'''
    for mcCycles in numberOfLoops:
        
        # Start Metropolis algorithm with Monte Carlo sampling
        i = 0
        for temp in range(int(tempInit/tempStep),int(tempFinal/tempStep)+1,1):
            temp = temp*tempStep  
            expectationValues = zeros(6);
            t0 = time.clock()
            metropolis(nSpins, mcCycles, temp, expectationValues, acceptedFlips,True)
            t1 = time.clock()
            tim = t1-t0
            #print 'Time:',tim,'s'
            calcWrite(nSpins, mcCycles, temp, expectationValues, CvError, XError, acceptedFlips,mcCycles);
            print '''{:<25.1e}|{:<16.5e}|{:<17.5e}|{:>10.6}'''.format(mcCycles,CvError_list[-1],XError_list[-1],tim)
            i+=1    
            #print 'Accepted flips: ', acceptedFlips
        #print expectVal_E_list
        #energy.append(expectVal_E_list[0])
    figure(1)
    semilogx(numberOfLoops,expectVal_E_list)
    grid(True)
    xlabel(r'$number\ of\ cycles$')
    ylabel(r'${<E>}/{L^2}$')
    figure(2)
    semilogx(numberOfLoops,expectVal_M_list)
    grid(True)
    xlabel(r'$number\ of\ cycles$')
    ylabel(r'${<|M|>}/{L^2}$')
    figure(3)
    hist(expectVal_X_list)
    xlabel(r'$Energy$')
    ylabel(r'$Value$')


show()
