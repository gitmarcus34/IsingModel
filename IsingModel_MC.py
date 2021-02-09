"""This is my new saved program.
"""
# Control the way that the plots show up
#matplotlib inline

# Import the plotting commands
import matplotlib
import matplotlib.pyplot as plt

#gif writing software
from array2gif import write_gif

# We will use numpy for arrays because the python lists are kind of obnoxious
# numpy also has built in functionality for lots of things, like random numbers
import numpy as np

import time


"""
Note:

A lattice is a network of dipoles. Each dipole will interact with specific neighbors according to the Ising Model.

mDim is an integer setting for the number of dimensions of our lattice (ie the number of coordinates mapping each element/dipole in our lattice).
nSide is an integer setting for the number of dipoles that stretch along each coordinate axis of our lattice.

To implement the Ising Model we use the Monte Carlo algorithm.
"""

#general methods & functions
def createAlignedState(nSide,mDim):
	#a single row vector of mDim^d ones
	return np.ones(nSide**mDim)

def createRandomState(nSide, mDim):
	#a row vector of N^d random -1 or 1's
	return np.random.choice([-1,1], size = [nSide**mDim])

def coordToIndex(coord, nSide):
	"""Use the arity of an nSided lattice to map coordinates to index. 
		>>>Returns this index.
		
		- Coord is an m-dimensional coordinate (m being an arbitrary integer).
		- nSide is the number of elements along each coordinate axis of the lattice
		  (n being an arbitrary integer).

		We can take for example a 5 dimensional coordinate in a 16 sided lattice
		such as [1,5,13, 8, 0] and convert it from the hexadecimel number 157D80 to the corresponding
		decimel number-index (1*16^4 + 5*16^3 + 13*16^2 + 8*16^1 + 0*16^0) = 89747. 
		
		Note that we index from 0 which is the origin coordinate [0,0,...,0].	
	"""
	index = 0
	mDim = len(coord)
	for i in range(mDim):
		index += coord[i]*nSide**(mDim-1-i)
	return index
	
def indexToCoord(index, nSide, mDim):
	"""Use the arity of an nSided lattice to map index to its coorisponding coordinate in the lattice.
		>>>returns this coordinate
	
		- nSide is the number of elements along each coordinate axis of the lattice
		  (n being an arbitrary integer).
		-mDim is the number of coordinates used to map each point in the lattice.
		-Index is the numeric-decimel location of each element counting from 0 at the origin [0,0,...,0]; to
		1 at [0,0,...,1]; ... to nSide at [0,0,...,nSide]; to nSide+1 at [0,0,...1, nSide]; ... 
		to the final index nSide*nSide at [nSide, nSide,...nSide]
	
		We can convert the index utilizing the arithmetic of converting from the given decimel-index to the
		corresponding arity number where each digit of this number will be a corresponding coordinate of the 
		element.
		
		
	"""
	coord = []
	for i in range(mDim):
		coord.append((index % nSide**(mDim-i)) // nSide**(mDim-1-i))
	return coord

def indexToCoord2(index, nSide, mDim):
	coord = []
	for i in range(mDim):
		coord.append(index // nSide**(mDim-1-i))
		index = index % nSide**(mDim-1-i)
	return coord

def calcNeighborList(index, nSide, mDim):
	"""return a list of neighbors' indexes for a given element at index in an n-sided lattice with m-dimensions.
	
		Neighbors will be defined as those elements that are 1 unit in both directions along every axis.
		The elements at the boundaries of our lattice will be connected with elements on opposite ends
		of the lattice so that each element has an equal number of neighbors (for large lattices
		this will have insignifcant effects, so be weary using small lattices.)
		
		For example, in a 1-D lattice each element will have neighbors to the left and right, while
		the beginning and the end of the lattice will also be neighbors (like a circle).
	
	"""
	coord = indexToCoord(index, nSide, mDim)
	nList = []
	for i in range(mDim):
		for shift in [-1,1]:
			neighborCoord = coord.copy()
			neighborCoord[i] = (neighborCoord[i]+shift)%nSide
			nList.append(coordToIndex(neighborCoord, nSide))
	return nList

def createAdjMatrix(nSide, mDim):
	"""Create a matrix whose rows represent each element-index in our lattice and each column element in the row
	   shall represent each neighbor to the row element where each column corresponds to the element index of the lattice.  
	   If the column element is 0 then it is not a neighbor to the row element and if the column element is a 1 then it will 
	   be defined as an adjacency neighbor to the row element.
	   
	   For example, say we have a 2 sided, 3 dimensional lattice (note this will be a 2x2x2 cube lattice containing 8 total elements). 
	   Then, pick the element who is index in the lattice is given by the number 3.  As we can see in the matrix below, index 3 has 
	   three adjacent neighbors who are given by the indexes 1, 2, and 7.  
	   
	   Adjacency matrix of a 3 dimensional lattice with 2 elements on a side:
	   index 0: [[0., 1., 1., 0., 1., 0., 0., 0.],
	   index 1: [1., 0., 0., 1., 0., 1., 0., 0.],
	   index 2: [1., 0., 0., 1., 0., 0., 1., 0.],
	   index 3: [0., 1., 1., 0., 0., 0., 0., 1.],
	   index 4: [1., 0., 0., 0., 0., 1., 1., 0.],
	   index 5: [0., 1., 0., 0., 1., 0., 0., 1.],
	   index 6: [0., 0., 1., 0., 1., 0., 0., 1.],
	   index 7: [0., 0., 0., 1., 0., 1., 1., 0.]]
	   
	  Adjaceny matrices become useful when calculating the total energy of a lattice state.
	"""
	adjMatrix = np.zeros([nSide**mDim, nSide**mDim])
	for index in range(nSide**mDim):
		nList = calcNeighborList(index, nSide, mDim)
		adjMatrix[index, nList] = 1
	return adjMatrix

def calcTotalEnergy(state,adj):
	"""Return the total energy of the system according to Ising Model
	   Energy = sum(state[i]*state[j]) from i,j = (0,0) to (nSide-1,nSide-1)
	   
	   We can use matrix multiplication to multiply state by its given adjacency matrix.  This gives
	   a vector with nSide elements. Then we take the dot product of this vector with the state.
	   
	   For example, given a 3 dimensional state with 2 on a side we see the following operation take place
										
										Adj matrix
								   [[0. 1. 1. 0. 1. 0. 0. 0.]
					State			[1. 0. 0. 1. 0. 1. 0. 0.]
									[1. 0. 0. 1. 0. 0. 1. 0.]	  state X adjacency matrix
		[ 1  1  1  1  1  1 -1 -1] X [0. 1. 1. 0. 0. 0. 0. 1.] = [3. 3. 1. 1. 1. 1. 1. 1.]
									[1. 0. 0. 0. 0. 1. 1. 0.]
									[0. 1. 0. 0. 1. 0. 0. 1.]
									[0. 0. 1. 0. 1. 0. 0. 1.]
									[0. 0. 0. 1. 0. 1. 1. 0.]]
									
		(state X adjacency matrix)  (state transpose) 
		[3. 3. 1. 1. 1. 1. 1. 1.] X [1  
									 1  
									 1	Total Energy
									 1  = 8.0
									 1  
									 1 
									-1 
									-1]]

		The first matrix multiplication essentially seeks out the number of neighbors for each
		element in the state.
		The second matrix multiplication essentially imposes the negative signs in the correct positions
		since the adjacency matrix lacks this information given all ones are positive.
		
	#this is State * Adjacency Matrix * State transpose.
	#simple matrix multiplication finds the energy, since the adjacency neighbor connects neighbors
	"""
	#this is State * Adjacency Matrix * State transpose.
	#simple matrix multiplication finds the energy, since the adjacency neighbor connects neighbors
	return -np.dot(np.dot(state,adj),state)/2

def pickRandomSite(nSide,mDim):
	""">>>returns a random integer which will be a random dipole-index for a given lattice with nSide and mDim.
	"""
	return np.random.randint(nSide**mDim)

def calcDeltaE(state,adj,site):
	"""State is the current lattice state, adj is the adjacency matrix for the lattice, and site will be a given dipole in the lattice
		>>>returns a float value which represents the energy of a site and its neighbors.
	
		This function is primarily used in the isingND function where site will be a randomly selected dipole in the given lattice. 
		
		deltaE (change in energy) is calculated after a dipole is either flipped or not flipped through each pass in the Monte Carlo algorithm.  
		The following things can happen between the site and one of its neighbors each time we pass through the algorithm:
		
		1. The site is flipped from -1 to 1
		initial_energy = (-1)*(1)
		new_energy = (1)*(1)
		deltaE = new_energy - initial_energy = (1) - (-1) = 2 
		
		2. The site is flipped from (1) to (-1)
		initial_energy = (1)*(1)
		new_energy = (-1)*(1)
		deltaE = new_energy - initial_energy = (-1) - (1) = -2 
		
		3. The site is NOT flipped at all
		initial_energy = (1) or (-1)
		new_energy = (1) 0r (-1)
		deltaE = (1)-(1) = 0 or deltaE = (-1)-(-1) = 0
		
		Now note if the site has an even number of neighbors then the following instances can happen:
		
		1.The initial_energy = 0, 
		flipping the site will result in new_energy = 0 due to symmetry.  Therefore, the deltaE = 0
		Example consider the middle site in the following: 
		(1)(-1)(-1) -> initial_energy = (-1)*(1) + (-1)*(-1) = 0 
		(1)(1)(-1) -> new_energy = (1)*(1) + (1)*(-1) = 0
		
		2. The initial energy is positive or negative:
		flipping the site will result in new_energy = -1 * initial_energy. Therefore, deltaE = 2*|initial_energy|
		
		
		
		Essentially the math that follows takes advantage of distribution in the following way:
		(site*neighbor1 + site*neighbor1 + site*neighbor1 + ... + site*neighborN) = site*(neighbor1 + neighbor2 + ... + neighborN)
		
		We collect the neighbors by multiplying state by the corresponding site's row of neighbors in the adjacency matrix.
		We sum the neighbors together and multiply the sum it by 2*site.  The multiplication of 2 is included because flipping
		the dipole corresponds to an energy change of 2 units since |(1) - (-1)| = 2.			
	"""
	neighbors=adj[site,:]
	deltaE=2*state[site]*sum(state*neighbors)
	return deltaE
	
	
def isingND(state, adjacencyMatrix, nSide, mDim, temperature, nSteps):
	"""Implements monte carlos algorithm to simulate the Ising Model.
	   
	   1. Pick a random site
	   2. Calculate the change in energy from the previous state to the new state.
	   3. Set the probability to change or maintain the random site we chose.
		  -If change in energy between previous state and new state (i.e. deltaE) is less than 0 then the probability to flip is 100%
		  -If there was no change in energy then the probability to flip will be 1/2
		  -If deltaE is greater than 0 and temperature is 0 then the probability to flip is 0%
		   Otherwise set the probability to flip to e^(-deltaE/T)
	   4.  Pick a random number between 0 and 1 and if this number is less or equal to our probability to flip then flip the site.
		   Otherwise keep the lattice state the same.
		
		During this algorithm we collect in arrays each change in energy and each energy state after each change in temperature implementation.
		These two arrays as well as the final state are returned.
	"""   
	deltaE_list = []
	E = np.zeros(nSteps)
	E[0] = calcTotalEnergy(state,adjacencyMatrix)
			
	for t in range(1,nSteps):
		site = pickRandomSite(nSide,mDim)		
		deltaE = calcDeltaE(state, adjacencyMatrix, site)
		deltaE_list.append(deltaE)
		#calculate the probability of flipping:
		if(deltaE<0):
			probabilityToFlip=1
		elif deltaE==0:
			probabilityToFlip=1/2
		#deltaE>0
		else:
			if temperature==0:
				probabilityToFlip=0
			else:
				probabilityToFlip=np.exp(-deltaE/temperature)
			
		#generate a random number, and use it to decide to flip the spin
		#here we are avoiding recalculating the energy at each time step!
		if(np.random.rand()<=probabilityToFlip):			
			state[site]*=-1
			E[t] = E[t-1] + deltaE
		else:
			E[t] = E[t-1]
			
	return state, E, deltaE_list
	
#Function that implement plots mean energy, mean magnetization, heat capacity and magnetic susceptibility each with respect to temperature
#Temperature increments by 0.1 units per Ising implementation
def meanE_meanM_heatCap_magSus(nSide, mDim, T_init, T_final, nSteps):
	"""repeatedly implements the Ising Model simulation over a range of temperatures n-Sided and m-dimensional lattices which start with 
	   random configurations.  When each implementation is complete we collect the final states and energies of the lattices and calculate
	   the mean magnetization, mean energy, magnetic susceptibility, and heat capacity.
	
	"""
	adj = createAdjMatrix(nSide, mDim)
	meanEnergy = []
	meanMag = []
	heatCap = []
	magSus = []
	
	temp_list = []
	temp = T_init
	
	while temp < T_final:
		state=createAlignedState(nSide, mDim)
		#implement the Monte Carlos algorithm
		finalState,E, deltaE=isingND(state, adj, nSide, mDim, temp, nSteps)

		#calculate the average energy, average magnetization, heat capacity, and magnetic suscpetibility at the end of
		#each Ising model implementation
		meanEnergy.append(np.mean(E))
		
		#mean is same as sum of each spin state divided by the total number of states: (1D = 10, 2D = 100, 3D= 1000, 4D = 10000)
		meanMag.append(np.mean(finalState))
		
		#meanCap is variance of the energy states
		heatCap.append(np.var(E))
		
		#meanMag is variance of the magnetization state
		magSus.append(np.var(finalState))

		#collect each incremented temperature used for each implementation of the code
		temp_list.append(temp)
		temp = temp + 0.1
		
	#plot mean energy vs time
	plt.plot(temp_list, meanEnergy, 'bo')
	plt.title("Mean-Energy vs Temperature")
	plt.xlabel('Temperature')
	plt.ylabel('Mean-Energy')
	plt.show()

	#plot mean magnetization vs temperature
	plt.plot(temp_list, meanMag, 'ro')
	plt.title("Mean-Magnetization vs Temperature")
	plt.xlabel('Temperature')
	plt.ylabel('Mean-Magnetization')
	plt.show()
	
	
	#plot heat-capacity vs temperature
	plt.plot(temp_list, heatCap, 'go')
	plt.title("Heat-Capacity vs Temperature")
	plt.xlabel('Temperature')
	plt.ylabel('Heat-Capacity')
	plt.show()
	
	#plot magnetic susceptibility vs temperature
	plt.plot(temp_list, magSus, 'ro')
	plt.title("Magnetic Susceptibility vs Temperature")
	plt.xlabel('Temperature')
	plt.ylabel('Magnetic Susceptibility')
	plt.show()
	
	
	return meanEnergy, meanMag, heatCap, magSus
	
	
def animateIsing2D_Data(nSteps, nSide, temperature, number_of_images):
	"""create an animation of the ising model
	"""
	state = createRandomState(nSide, 2)
	adjacencyMatrix = createAdjMatrix(nSide, 2)
	
	
	deltaE_list = []
	E = np.zeros(nSteps)
	E[0] = calcTotalEnergy(state,adjacencyMatrix) #first entry will be the total energy of the intital state
	state_data = []
	
	for t in range(1,nSteps):
		site = pickRandomSite(nSide,2)		
		deltaE = calcDeltaE(state, adjacencyMatrix, site)
		deltaE_list.append(deltaE)
		
		
		if t%(nSteps/number_of_images) == 0:
			state_data.append(state)

		#calculate the probability of flipping:
		if(deltaE<0):
			probabilityToFlip=1
		elif deltaE==0:
			probabilityToFlip=1/2
		#deltaE>0
		else:
			if temperature==0:
				probabilityToFlip=0
			else:
				probabilityToFlip=np.exp(-deltaE/temperature)
			
		#generate a random number, and use it to decide to flip the spin
		#here we are avoiding recalculating the energy at each time step!
		if(np.random.rand()<=probabilityToFlip):			
			state[site]*=-1
			E[t] = E[t-1] + deltaE
		else:
			E[t] = E[t-1]
	
	#print("state data {}".format(state_data))
	return state_data
	

def dipoles_pixelsConvert(state):
	negPixel = [0,0,0]
	posPixel = [255,255,255]
	
	pixel_state = []
	for element in state:
		if element == -1:
			pixel_state.append(negPixel)
		elif element == 1:
			pixel_state.append(posPixel)
	return pixel_state
	
def convert2DState_Pixel(state_data, nSide):
	
	dataset = []
	
	for state in state_data:
		pixel_state = dipoles_pixelsConvert(state)
		pixel_array = np.array(pixel_state)
		pixel_array.shape = (nSide, nSide, 3)
		
		
		dataset.append(pixel_array)
	#print("dataset {}".format(dataset))
		
	return dataset
		
def main():
	nSide = 10
	mDim = 2
	temperature = 1
	nSteps = 1000
	
	state = createRandomState(nSide, mDim)
	adj = createAdjMatrix(nSide, mDim)
	
	number_of_images = 4
	state_data = animateIsing2D_Data(nSteps, nSide, temperature, number_of_images)
	
	dataset = convert2DState_Pixel(state_data, nSide)
	print("Dataset: {}".format(dataset))
	write_gif(dataset, 'rgbbgr.gif', fps=5)
		
if __name__ != "__main__":
    main()
