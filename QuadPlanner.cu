#include <math.h>
#include "Structs.h"
#include <vector>
#include <stdlib.h>
#include <stdio.h>
#include "hash_map_template.h"
#include "cuda_profiler_api.h"
 
#define	STARTING_VALUE -1
#define OBSTACLE_VALUE -2
#define GOAL_VALUE -3
#define NUM_NEIGH_PER_QUAD	20	
#define BLOCK_SIZE 512
//Defines Texture and its methods

using namespace CUDASTL;

HashMap<int, int>* hashmap;
int* neighborIndexesDev;

__device__ void neighborsForQuadDev(int* neighbors, QuadStruct* quad, HashMap<int, int> *hashmap);

__device__ bool equals(float x1, float x2)
{
	if (fabs(x1 - x2) < .01) {
		return true;
	}
	return false;
}

__device__ float distance(QuadStruct *from, QuadStruct *to) {
	
	if (equals(from->centerx,to->centerx)) {
		return fabs(to->centery - from->centery);
	}
	else if(equals(from->centery,to->centery)) {
		return fabs(to->centerx - from->centerx);
	}
	else {
		return sqrt(fabs(to->centerx - from->centerx)*fabs(to->centerx - from->centerx) +
			fabs(to->centery - from->centery)*fabs(to->centery - from->centery));
	}
}

__device__ int stateNeedsUpdate(QuadStruct* state) {
	return state->g == STARTING_VALUE || state->g == GOAL_VALUE;
}

__device__ int stateIsObstacle(QuadStruct* state) {
	return state->costToReach > 10.0f;
}

__device__ int QisGoalState(QuadStruct* state) {
	return state->g == 0.0f;
}

//Kernel function for planner

__global__ void computeNeighborsKernel(QuadStruct *current_texture, HashMap<int, int> *hashmap, int numberOfQuads, int* neighborIndexes)
{
	int id = get_thread_id();
	if (id < numberOfQuads) {
		QuadStruct quad = current_texture[id];
		neighborsForQuadDev(&neighborIndexes[id*NUM_NEIGH_PER_QUAD], &quad, hashmap);
		current_texture[id] = quad;
	}
}

extern "C" void computeNeighbors(QuadStruct* texture, int numberOfQuads)
{
	int gridLength = ceil((double)numberOfQuads/(double)BLOCK_SIZE);
	
	dim3 blocks(gridLength, 1, 1);
	dim3 threads(BLOCK_SIZE, 1, 1);
	
	QuadStruct *texture_device;
	cudaMalloc((void**)&texture_device, (numberOfQuads)*sizeof(QuadStruct));
	
	//make a two copies of the initial map
	cudaMemcpy(texture_device, texture, (numberOfQuads)*sizeof(QuadStruct), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&neighborIndexesDev, (numberOfQuads*NUM_NEIGH_PER_QUAD)*sizeof(int));
	cudaMemset(neighborIndexesDev, -1, (numberOfQuads*NUM_NEIGH_PER_QUAD)*sizeof(int));

	computeNeighborsKernel<<<blocks, threads>>>(texture_device, hashmap, numberOfQuads, neighborIndexesDev);

	cudaMemcpy(texture, texture_device, (numberOfQuads)*sizeof(QuadStruct), cudaMemcpyDeviceToHost);
	cudaFree(texture_device);
}

__global__ void QcomputeCostsKernel(QuadStruct *current_texture, QuadStruct *texture_copy, int* neighborIndexes, HashMap<int, int> *hashmap, int numberOfQuads, int *check, int *locality, float maxCost, bool allAgentsReached) {
	int id = get_thread_id();

	if (id < numberOfQuads) {
		QuadStruct quad = current_texture[id];

		//if(!stateIsObstacle(state) && !isGoalState(state)) {
			//if the state is an obstacle, do not compute neighbors
		if (!QisGoalState(&quad)) {

			int predecesorIndex;
			for (int i = 0; i < NUM_NEIGH_PER_QUAD; ++i) {
				int neighborIndex = neighborIndexes[(id*NUM_NEIGH_PER_QUAD)+i];
				if (neighborIndex < 0) 
					break;
				QuadStruct neighbor = texture_copy[neighborIndex]; //Needs to find a quad in the ro map

				if (stateIsObstacle(&neighbor)) //if neighbor is an obstacle, do not use it as a possible neighbor
					continue;
				float newg = neighbor.g + distance(&neighbor, &quad) * quad.costToReach;
				if ((newg < quad.g || stateNeedsUpdate(&quad)) && !stateNeedsUpdate(&neighbor)) {
					predecesorIndex = neighbor.indexInMap;
					quad.prevQuadCode = neighbor.quadCode;
					quad.g = newg;
					if (*locality == 1) {
						*check = 0;
					} else if (*locality == 2) {
						if (quad.g < maxCost || !allAgentsReached) {
							*check = 0;
						}
					} else if (*locality == 0 && allAgentsReached) {
						*check = 1;
					}
				}
			}
			
		/*	QuadStruct *selectedPredecessorCopy = &texture_copy[predecesorIndex];
			quad->inconsistent = false;
			//if ((selectedPredecessorCopy != NULL && selectedPredecessorCopy->inconsistent) || stateIsObstacle(selectedPredecessorCopy)) {
			if (selectedPredecessorCopy->inconsistent) {
				//if predecessor from read-only is inconsistent - clear inconsistent flag in write-only and mark state as inconsistent in write-only
				current_texture[predecesorIndex].inconsistent = false;
				quad->inconsistent = true;
				quad->g = STARTING_VALUE;
			} */
		}
		current_texture[id] = quad;
	}	
}

__global__ void checkForInconsistency(QuadStruct* texture, int numberOfQuads, int* flag) {
	int id = get_thread_id();

	if (id < numberOfQuads) {
		QuadStruct* state = &texture[id];
		if (state->inconsistent) {
			*flag = 1;
		}
	}
}

float agentsMaxCost(QuadStruct* texture, int agentCount, QuadStruct* agents, int goalNumber) {
	float maxCost = -10000.0f;
	for (int i = 0; i < agentCount; i++)  {
		QuadStruct agent = texture[agents[i].indexInMap];
		if (agent.g > maxCost) {
			maxCost = agent.g;
		}
	}
	return maxCost;
}

bool agentsReached(QuadStruct* texture, int agentCount, QuadStruct* agents, int goalNumber) {
	for (int i = 0; i < agentCount; i++) {
		QuadStruct agent = texture[agents[i].indexInMap];
		if (agent.g < 0.0f) {
			return false;
		}
	}
	return true;
}


extern "C" int QcomputeCostsCuda(QuadStruct* texture, int numberOfQuads, int locality, int agentCount, QuadStruct* agents, int goalNumber) {
	int *locality_dev, *consistencyCheck, *consistencyCheck_dev, *flag, *flag_dev;
	
	int gridLength = ceil((double)numberOfQuads/(double)BLOCK_SIZE);
	
	dim3 blocks(gridLength, 1, 1);
	dim3 threads(BLOCK_SIZE, 1, 1);

	QuadStruct *texture_device, *texture_device_copy;
	cudaMalloc((void**)&texture_device, (numberOfQuads)*sizeof(QuadStruct));
	cudaMalloc((void**)&texture_device_copy, (numberOfQuads)*sizeof(QuadStruct));
	//make a two copies of the initial map
	cudaMemcpy(texture_device, texture, (numberOfQuads)*sizeof(QuadStruct), cudaMemcpyHostToDevice);
	cudaMemcpy(texture_device_copy, texture, (numberOfQuads)*sizeof(QuadStruct), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&locality_dev, sizeof(int));
	int* locality_ptr = (int*)malloc(sizeof(int));
	*locality_ptr = locality;

	cudaMalloc((void**)&consistencyCheck_dev, sizeof(int));
	consistencyCheck = (int*)malloc(sizeof(int));

	cudaMalloc((void**)&flag_dev, sizeof(int));
	flag = (int*)malloc(sizeof(int));

	int iterations = 0;

	do {
		//set flag to 0 to check for changes
		if (locality == 1 || locality == 2) {
			*consistencyCheck = 1;
		} else {
			*consistencyCheck = 0;
		}
		*flag = 0;
		cudaMemcpy(locality_dev, locality_ptr, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(consistencyCheck_dev, consistencyCheck, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(flag_dev, flag, sizeof(int), cudaMemcpyHostToDevice);

		bool allAgentsReached = agentsReached(texture, agentCount, agents, goalNumber);
		float maxCost;
		if (allAgentsReached) {
			maxCost = agentsMaxCost(texture, agentCount, agents, goalNumber);
		}
		QcomputeCostsKernel<<<blocks, threads>>>(texture_device, texture_device_copy, neighborIndexesDev, hashmap, numberOfQuads, consistencyCheck_dev, locality_dev, maxCost, allAgentsReached);
		
		checkForInconsistency<<<blocks, threads>>>(texture_device, numberOfQuads, flag_dev);
		
		QuadStruct* temp = texture_device;
		texture_device = texture_device_copy;
		texture_device_copy = temp;
		iterations++;
		
		cudaMemcpy(texture, texture_device, (numberOfQuads)*sizeof(QuadStruct), cudaMemcpyDeviceToHost);
		
		cudaMemcpy(consistencyCheck, consistencyCheck_dev, sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(flag, flag_dev, sizeof(int), cudaMemcpyDeviceToHost);
	} while(*consistencyCheck == 0 || *flag == 1);

	
	cudaFree(texture_device); cudaFree(texture_device_copy);

	printf("Result was: %i\n\n", *consistencyCheck);
	printf("Number of iterations: %i\n\n", iterations);

	return 1;
}

__global__ void clearTextureValuesKernel(QuadStruct* texture, int numberOfQuads, int goalNumber) {
	int id = get_thread_id();

	if (id < numberOfQuads) {
		QuadStruct* state = &texture[id];
		state->g = STARTING_VALUE;
		state->prevQuadCode = 0;
		state->inconsistent = false;
	}

}

extern "C" void clearTextureValuesQuad(QuadStruct* texture, int numberOfQuads, int goalNumber) {
	QuadStruct* texture_dev;

	int gridLength = ceil((double)numberOfQuads/(double)BLOCK_SIZE);
	dim3 blocks(gridLength, 1, 1);
	dim3 threads(BLOCK_SIZE, 1, 1);

	cudaMalloc((void**)&texture_dev, (numberOfQuads*sizeof(QuadStruct)));
	cudaMemcpy(texture_dev, texture, numberOfQuads*sizeof(QuadStruct), cudaMemcpyHostToDevice);
	clearTextureValuesKernel<<<blocks, threads>>> (texture_dev, numberOfQuads, goalNumber);
	cudaMemcpy(texture, texture_dev, (numberOfQuads)*sizeof(QuadStruct), cudaMemcpyDeviceToHost);

	cudaFree(texture_dev);
}

__global__ void populateHashMap(HashMap<int, int> *hash,  QuadStruct *quads, int numberOfQuads)
{
	int i = get_thread_id();
	if (i < numberOfQuads) {
		(*hash)[quads[i].quadCode] = quads[i].indexInMap;
	}
}

extern "C" void createHashMap(QuadStruct quads[], int numberOfQuads, int size)
{
	hashmap = CreateHashMap<int,int, HashFunc<int> >(ceil((double)numberOfQuads/32), size);
	cudaError err = cudaGetLastError();
	int blocks = ceil((double)numberOfQuads/512);

	QuadStruct* quads_dev, *q;
	q = (QuadStruct*)malloc(sizeof(QuadStruct));
	q = quads;

	cudaMalloc((void **)&quads_dev, sizeof(QuadStruct)*numberOfQuads);
	cudaMemcpy(quads_dev, q, sizeof(QuadStruct)*numberOfQuads, cudaMemcpyHostToDevice);

	populateHashMap<<<blocks, 512>>> (hashmap, quads_dev, numberOfQuads);
	cudaFree(quads_dev);
}

__global__ void retrieveQuadStruct(int* quad, int code, HashMap<int, int> *hashmap)
{
	int i = get_thread_id();
	if (i == 1) {
		int q = *((*hashmap).valueForKey(code));
		*quad = q; 
	}
}

extern "C" int *quadForCode(int code)
{
	int *q_dev, *q;
	q = (int*)malloc(sizeof(int));
	cudaMalloc((void**)&q_dev, sizeof(int));
	
	retrieveQuadStruct<<<1, 512>>> (q_dev, code, hashmap);
	cudaMemcpy(q, q_dev, sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(q_dev);
	return q;
}

extern "C" void cleanupDevice()
{
	DestroyHashMap(hashmap);
	cudaDeviceReset();
}






/**************************************************************
******************** Neighbor Finding *************************
***************************************************************/

__device__ int constructNeighborQuadCode(QuadStruct* quad, int max_i, int codeDiff)
{		
	int code = 0;
	int codeDigit = quad->quadCode;
	int compareInt = 10, substract;

	int length = 1;
	while (codeDigit >= compareInt) {
		codeDigit /= 10;
		length++;
	}

	substract = codeDigit;

	for (int i = 0; i < length; i++) 
	{
		if (i+1 > max_i) {
			codeDigit -= codeDiff;	
		} else if (i+1 == max_i) {
			codeDigit += codeDiff;
		}

		code = (code*10) + codeDigit;

		compareInt *= 10;
		codeDigit = quad->quadCode;

		while (codeDigit >= compareInt) {
			codeDigit /= 10;
		}

		codeDigit -= substract*10;
		substract = substract*10+codeDigit;
	}
	return code;
}

__device__ int* greaterNeighbor(QuadStruct quad, int neighborQuadCode, HashMap<int, int> *hashmap)
{
	int parsedCode = neighborQuadCode;

	int* neighbor = NULL;
	while (parsedCode > 0)
	{
		if (parsedCode == neighborQuadCode) {
			neighbor = (*hashmap).valueForKey(parsedCode);
			if (neighbor != NULL) {
				break;
			}
		}

		parsedCode = parsedCode/10;
		neighborQuadCode = neighborQuadCode/10;
	}

	return neighbor;
}

__device__ void populateSmallerNeighbors(QuadStruct* quad, int* neighbors, int quadCode, int addCode1, int addCode2, HashMap<int, int> *hashmap, int *indexOffset)
{
	int* neighbor = (*hashmap).valueForKey(quadCode);

	if (neighbor != NULL) {
		int q = *neighbor;
		neighbors[*indexOffset] = q;
		*indexOffset += 1;
		quad->neighborCount += 1;
	} else {
		int *codes = (int*)malloc(sizeof(int)*(16));
		codes[0] = (quadCode*10) + addCode1;
		codes[1] = (quadCode*10) + addCode2;
		int lastIndex = 2;
		for (int i = 0; i < lastIndex; i++) {
			int code = codes[i];
			if (code/100000 > 0) { continue; }

			int* subNeighbor = (*hashmap).valueForKey(code);
			if (subNeighbor != NULL) {
				int q = *subNeighbor;
				neighbors[*indexOffset] = q;
				*indexOffset += 1;
				quad->neighborCount += 1;
			} else {
				codes[lastIndex] = (code*10) + addCode1;
				codes[lastIndex+1] = (code*10) + addCode2;
				lastIndex += 2;
			}
		}
	}
}

__device__ int* smallerNeighbors(QuadStruct* quad, int neighborQuadCode, int addCode1, int addCode2, HashMap<int, int> *hashmap)
{
	int* neighbors = (int*)malloc(sizeof(int)*16);

	int neighborCode1 = (neighborQuadCode*10) + addCode1;
	int neighborCode2 = (neighborQuadCode*10) + addCode2;

	int currentNeighbors = quad->neighborCount;
	int *indexOffset = (int*)malloc(sizeof(int));
	*indexOffset = 0;
	populateSmallerNeighbors(quad, neighbors, neighborCode1, addCode1, addCode2, hashmap, indexOffset);

	populateSmallerNeighbors(quad, neighbors, neighborCode2, addCode1, addCode2, hashmap, indexOffset);

	return neighbors;
}

__device__ void retrieveNeighbors(int* neighbors, QuadStruct* quad, int max_i, HashMap<int, int> *hashmap, int codeDiff, int addCode1, int addCode2)
{
	int neighborCode = constructNeighborQuadCode(quad, max_i, codeDiff);
	int* neighborQuads = (*hashmap).valueForKey(neighborCode);
	int startingNeighborCount = quad->neighborCount;
	bool freeMem = false;
	if (neighborQuads == NULL) {
		neighborQuads = greaterNeighbor(*quad, neighborCode, hashmap);
		if (neighborQuads == NULL) {
			neighborQuads = smallerNeighbors(quad, neighborCode, addCode1, addCode2, hashmap);
			freeMem = true;
		} else {
			quad->neighborCount += 1;
		}
	} else {

		quad->neighborCount += 1;
	}

	int diff = quad->neighborCount - startingNeighborCount;
	for (int i = 0; i < quad->neighborCount; i++)
	{
		if (neighbors[i] < 0) {
			neighbors[i] = neighborQuads[diff-1];
			diff--;
			if (diff == 0) { break;}
		}
	}
	if (freeMem) {
		free(neighborQuads);
	}
}

__device__ void neighborsForQuadDev(int* neighbors, QuadStruct* quad, HashMap<int, int> *hashmap)
{
	short int Emax_i = 0;  short int Wmax_i = 0; short int Nmax_i = 0; short int Smax_i = 0;
	short int length = 1;
	int codeDigit = quad->quadCode;
	int compareInt = 10;
	int substract = 0;
	quad->neighborCount = 0;
	
	while (codeDigit >= compareInt) {
			codeDigit /= 10;
			length++;
	}


	for (int i = 0; i < length; i++) 
	{
		while (codeDigit >= compareInt) {
			codeDigit /= 10;
		}
		codeDigit -= substract*10;
		substract = substract*10+codeDigit;

		if (codeDigit == 1 || codeDigit == 3) {
			Emax_i = i+1;	
		}

		if (codeDigit == 2 || codeDigit == 4) {
			Wmax_i = i+1;	
		}

		if (codeDigit == 1 || codeDigit == 2) {
			Smax_i = i+1;	
		}

		if (codeDigit == 3 || codeDigit == 4) {
			Nmax_i = i+1;	
		}

		compareInt *= 10;
		codeDigit = quad->quadCode;
	}

	if (Emax_i > 0) {
		retrieveNeighbors(neighbors, quad, Emax_i, hashmap, 1, 1, 3);
	}


	if (Wmax_i > 0) {
		retrieveNeighbors(neighbors, quad, Wmax_i, hashmap, -1, 2, 4);
	}


	if (Smax_i > 0) {
		retrieveNeighbors(neighbors, quad, Smax_i, hashmap, 2, 1, 2);
	}

	if (Nmax_i > 0) {
		retrieveNeighbors(neighbors, quad, Nmax_i, hashmap, -2, 3, 4);
	}
}