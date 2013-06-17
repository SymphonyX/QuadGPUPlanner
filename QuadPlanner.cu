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
#define INVALID_QUAD -5.0f
#define NUM_NEIGH_PER_QUAD	20	
#define BLOCK_SIZE 512
//Defines Texture and its methods

using namespace CUDASTL;

HashMap<int, int>* hashmap;
NeighborStruct* neighborsDev;

__device__ void neighborsForQuadDev(NeighborStruct* neighbors, QuadStruct* quad, HashMap<int, int> *hashmap);

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

__global__ void computeNeighborsKernel(QuadStruct *current_texture, HashMap<int, int> *hashmap, int numberOfQuads, NeighborStruct* neighbors)
{
	int id = get_thread_id();
	if (id < numberOfQuads) {
		QuadStruct quad = current_texture[id];
		if (quad.g != INVALID_QUAD) {
			neighborsForQuadDev(&neighbors[id*NUM_NEIGH_PER_QUAD], &quad, hashmap);
			current_texture[id] = quad;
		}
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

	cudaFree(neighborsDev);
	cudaMalloc((void**)&neighborsDev, (numberOfQuads*NUM_NEIGH_PER_QUAD)*sizeof(NeighborStruct));
	cudaMemset(neighborsDev, -1, (numberOfQuads*NUM_NEIGH_PER_QUAD)*sizeof(NeighborStruct));

	computeNeighborsKernel<<<blocks, threads>>>(texture_device, hashmap, numberOfQuads, neighborsDev);

	cudaMemcpy(texture, texture_device, (numberOfQuads)*sizeof(QuadStruct), cudaMemcpyDeviceToHost);
	cudaFree(texture_device);
}

__global__ void QcomputeCostsKernel(QuadStruct *current_texture, QuadStruct *texture_copy, NeighborStruct* neighbors, int numberOfQuads, int *check, int *locality, float maxCost, bool allAgentsReached) {
	int id = get_thread_id();

	if (id < numberOfQuads) {
		QuadStruct quad = current_texture[id];

		//if(!stateIsObstacle(state) && !isGoalState(state)) {
			//if the state is an obstacle, do not compute neighbors
		if (!QisGoalState(&quad) && quad.g != INVALID_QUAD) {

			int predecesorIndex;
			float originalG = quad.g;
			quad.g = STARTING_VALUE;
			for (int i = 0; i < NUM_NEIGH_PER_QUAD; ++i) {
				NeighborStruct neighbor_struct = neighbors[(id*NUM_NEIGH_PER_QUAD)+i];
				if (neighbor_struct.indexInMap < 0 ) 
					break;
				QuadStruct neighbor = texture_copy[neighbor_struct.indexInMap]; //Needs to find a quad in the ro map
				
				//if (neighbor.quadCode != neighbor_struct.quadCode) {
					//...Our index in map is outdated and neighbors need to be recomputed for this quad. Determine best way to go about it.
				//}

				if (stateIsObstacle(&neighbor)) //if neighbor is an obstacle, do not use it as a possible neighbor
					continue;
				float newg = neighbor.g + distance(&neighbor, &quad) * quad.costToReach;
				if ((newg < quad.g || stateNeedsUpdate(&quad)) && !stateNeedsUpdate(&neighbor)) {
					predecesorIndex = neighbor.indexInMap;
					quad.prevQuadCode = neighbor.quadCode;
					quad.g = newg;
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
			if (*locality == 1 && originalG != quad.g) {
				*check = 0;
			} else if (*locality == 2) {
				if ((originalG != quad.g && quad.g < maxCost) || !allAgentsReached) {
					*check = 0;
				}
			} else if (*locality == 0 && allAgentsReached && originalG != quad.g) {
				*check = 1;
			}
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

		QcomputeCostsKernel<<<blocks, threads>>>(texture_device, texture_device_copy, neighborsDev, numberOfQuads, consistencyCheck_dev, locality_dev, maxCost, allAgentsReached);
		
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

__global__ void propagateUpdateKernel(QuadStruct* texture, QuadStruct* texture_copy, int numberOfQuads, int* propagateUpdate, HashMap<int, int>* hashmap)
{
	int id = get_thread_id();
	if (id < numberOfQuads) {
		QuadStruct quad = texture_copy[id];
		if (quad.prevQuadCode > 0) {
			int predecessorIndex = *((*hashmap).valueForKey(quad.prevQuadCode));
			QuadStruct predecessor = texture[predecessorIndex];
				if (predecessor.g == STARTING_VALUE) {
					texture[id].g = STARTING_VALUE;
					texture[id].prevQuadCode = 0;
					*propagateUpdate = 1;
				}
			}
	}
}

extern "C" void propagateUpdateAfterObstacleMovement(QuadStruct* texture, int numberOfQuads)
{
	int gridLength = ceil((double)numberOfQuads/(double)BLOCK_SIZE);

	dim3 blocks(gridLength, 1, 1);
	dim3 threads(BLOCK_SIZE, 1, 1);

	QuadStruct* texture_dev, *texture_copy_dev;
	cudaMalloc((void**)&texture_dev, sizeof(QuadStruct)*numberOfQuads);
	cudaMalloc((void**)&texture_copy_dev, sizeof(QuadStruct)*numberOfQuads);

	cudaMemcpy(texture_dev, texture, sizeof(QuadStruct)*numberOfQuads, cudaMemcpyHostToDevice);
	cudaMemcpy(texture_copy_dev, texture, sizeof(QuadStruct)*numberOfQuads, cudaMemcpyHostToDevice);

	int* propagateUpdate = (int*)malloc(sizeof(int));
	
	int* propagateUpdate_dev;
	cudaMalloc((void**)&propagateUpdate_dev, sizeof(int));
	do  {
		*propagateUpdate = 0;
		cudaMemcpy(propagateUpdate_dev, propagateUpdate, sizeof(int), cudaMemcpyHostToDevice);
		
		propagateUpdateKernel<<<blocks, threads>>>(texture_dev, texture_copy_dev, numberOfQuads,propagateUpdate_dev, hashmap);
		cudaMemcpy(propagateUpdate, propagateUpdate_dev, sizeof(int), cudaMemcpyDeviceToHost);
		QuadStruct* temp = texture_dev;
		texture_dev = texture_copy_dev;
		texture_copy_dev = texture_dev;

	} while (*propagateUpdate == 1);
	
	cudaMemcpy(texture, texture_dev, sizeof(QuadStruct)*numberOfQuads, cudaMemcpyDeviceToHost);
	cudaFree(texture_dev); cudaFree(texture_copy_dev);
	
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



/************************************************
******** HashMap Methods ************************
************************************************/

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

__global__ void invalidateQuadsKernel(QuadStruct* quads, int* indexes, HashMap<int, int>* hashmap)
{
	int id = get_thread_id();
	//As many threads as objects to insert
	int* indexInMap = (*hashmap).valueForKey(quads[id].quadCode);
	int index = *indexInMap;
	indexes[id] = index;
	*indexInMap = -1;
}

extern "C" int* invalidateQuadsInHash(QuadStruct quadsRemoved[], int countRemoved)
{
	QuadStruct* ptr = quadsRemoved;
	QuadStruct* quads_dev;
	int* indexes = (int*)malloc(sizeof(int)*countRemoved);
	int* indexes_dev;

	cudaMalloc((void**)&quads_dev, sizeof(QuadStruct)*countRemoved);
	cudaMalloc((void**)&indexes_dev, sizeof(int)*countRemoved);

	cudaMemcpy(quads_dev, ptr, sizeof(QuadStruct)*countRemoved, cudaMemcpyHostToDevice);

	invalidateQuadsKernel<<<1, countRemoved>>>(quads_dev, indexes_dev, hashmap);

	cudaMemcpy(indexes, indexes_dev, sizeof(int)*countRemoved, cudaMemcpyDeviceToHost);

	cudaFree(indexes_dev); cudaFree(quads_dev);

	return indexes;
}

__global__ void updateQuadsKernel(QuadStruct* quads, int* indexes, HashMap<int, int>* hashmap)
{
	int id = get_thread_id();
	//As many threads as objects to insert
	int* indexInMap = (*hashmap).valueForKey(quads[id].quadCode);
	indexes[id] = *indexInMap;
}

extern "C" int* updateQuadsInHash(QuadStruct updateQuads[], int count)
{
	QuadStruct* ptr = updateQuads;
	QuadStruct* quads_dev;
	int* indexes = (int*)malloc(sizeof(int)*count);
	int* indexes_dev;

	cudaMalloc((void**)&quads_dev, sizeof(QuadStruct)*count);
	cudaMalloc((void**)&indexes_dev, sizeof(int)*count);

	cudaMemcpy(quads_dev, ptr, sizeof(QuadStruct)*count, cudaMemcpyHostToDevice);

	updateQuadsKernel<<<1, count>>>(quads_dev, indexes_dev, hashmap);

	cudaMemcpy(indexes, indexes_dev, sizeof(int)*count, cudaMemcpyDeviceToHost);

	cudaFree(indexes_dev); cudaFree(quads_dev);

	return indexes;
}

__global__ void insertNewQuadsKernel(QuadStruct quads[], HashMap<int, int>* hashmap)
{
	int id = get_thread_id();
	QuadStruct quad = quads[id];
	//As many threads as objects to insert
	int* indexInMap = (*hashmap).valueForKey(quad.quadCode);
	if (indexInMap == NULL) {
		(*hashmap)[quad.quadCode] = quad.indexInMap;
	} else {
		*indexInMap = quad.indexInMap;
	}
}

extern "C" void insertNewQuadsInHash(QuadStruct quadsInserted[], int countInserted)
{
	QuadStruct* ptr = quadsInserted;
	QuadStruct* quads_dev;

	cudaMalloc((void**)&quads_dev, sizeof(QuadStruct)*countInserted); 
	cudaMemcpy(quads_dev, ptr, sizeof(QuadStruct)*countInserted, cudaMemcpyHostToDevice);
	
	insertNewQuadsKernel<<<1, countInserted>>>(quads_dev, hashmap);

	cudaFree(quads_dev);
}


extern "C" void updateNeighborsToQuads(int* indexes, int size, int totalNumberOfQuads, QuadStruct** texture, int numberOfGoals, int* updateIndexes, int updateCount)
{
	NeighborStruct* neighbors = (NeighborStruct*) malloc ((totalNumberOfQuads*NUM_NEIGH_PER_QUAD)*sizeof(NeighborStruct));
	cudaMemcpy(neighbors, neighborsDev, (totalNumberOfQuads*NUM_NEIGH_PER_QUAD)*sizeof(NeighborStruct), cudaMemcpyDeviceToHost);

	int totalCount = size+updateCount; 
	for (int i = 0; i < totalCount; i++) {
		int offset = (i < size) ? indexes[i]*NUM_NEIGH_PER_QUAD : updateIndexes[i-size]*NUM_NEIGH_PER_QUAD;
		for (int j = 0; j < NUM_NEIGH_PER_QUAD; j++) {
			int neighborIndex = neighbors[offset+j].indexInMap;
			if (neighborIndex < 0) {
				continue;
			} else {
				for (int m = 0; m < numberOfGoals; m++) {
					texture[m][neighborIndex].g = STARTING_VALUE;
				}
			}
		}
	}
	free(neighbors);
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

__device__ int* greaterNeighbor(QuadStruct quad, int neighborQuadCode, HashMap<int, int> *hashmap, int* neighborCodes)
{
	int parsedCode = neighborQuadCode;

	int* neighbor = NULL;
	while (parsedCode > 0)
	{
		if (parsedCode == neighborQuadCode) {
			neighbor = (*hashmap).valueForKey(parsedCode);
			if (neighbor != NULL) {
				neighborCodes[0] = parsedCode;
				break;
			}
		}

		parsedCode = parsedCode/10;
		neighborQuadCode = neighborQuadCode/10;
	}

	return neighbor;
}

__device__ void populateSmallerNeighbors(QuadStruct* quad, int* neighbors, int quadCode, int addCode1, int addCode2, HashMap<int, int> *hashmap, int *indexOffset, int* neighborCodes)
{
	int* neighbor = (*hashmap).valueForKey(quadCode);

	if (neighbor != NULL && *neighbor != -1) {
		int q = *neighbor;
		neighbors[*indexOffset] = q;
		neighborCodes[*indexOffset] = quadCode;
		*indexOffset += 1;
		quad->neighborCount += 1;
	} else {
		int *codes = (int*)malloc(sizeof(int)*(16));
		codes[0] = (quadCode*10) + addCode1;
		codes[1] = (quadCode*10) + addCode2;
		int lastIndex = 2;
		for (int i = 0; i < lastIndex; i++) {
			int code = codes[i];
			if (code/1000000 > 5) { continue; }

			int* subNeighbor = (*hashmap).valueForKey(code);
			if (subNeighbor != NULL && *subNeighbor != -1) {
				int q = *subNeighbor;
				neighbors[*indexOffset] = q;
				neighborCodes[*indexOffset] = code;
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

__device__ int* smallerNeighbors(QuadStruct* quad, int neighborQuadCode, int addCode1, int addCode2, HashMap<int, int> *hashmap, int* neighborCodes)
{
	int* neighbors = (int*)malloc(sizeof(int)*16);

	int neighborCode1 = (neighborQuadCode*10) + addCode1;
	int neighborCode2 = (neighborQuadCode*10) + addCode2;

	int currentNeighbors = quad->neighborCount;
	int *indexOffset = (int*)malloc(sizeof(int));
	*indexOffset = 0;
	populateSmallerNeighbors(quad, neighbors, neighborCode1, addCode1, addCode2, hashmap, indexOffset, neighborCodes);

	populateSmallerNeighbors(quad, neighbors, neighborCode2, addCode1, addCode2, hashmap, indexOffset, neighborCodes);

	return neighbors;
}

__device__ void retrieveNeighbors(NeighborStruct* neighbors, QuadStruct* quad, int max_i, HashMap<int, int> *hashmap, int codeDiff, int addCode1, int addCode2)
{
	int neighborCode = constructNeighborQuadCode(quad, max_i, codeDiff);
	int* neighborQuads = (*hashmap).valueForKey(neighborCode);
	int* neighborCodes = (int*) malloc(sizeof(int)*8);
	int startingNeighborCount = quad->neighborCount;
	bool freeMem = false;
	if (neighborQuads == NULL || *neighborQuads == -1) {
		neighborQuads = greaterNeighbor(*quad, neighborCode, hashmap, neighborCodes);
		if (neighborQuads == NULL || *neighborQuads == -1) {
			neighborQuads = smallerNeighbors(quad, neighborCode, addCode1, addCode2, hashmap, neighborCodes);
			freeMem = true;
		} else {
			quad->neighborCount += 1;
		}
	} else {
		quad->neighborCount += 1;
		neighborCodes[0] = neighborCode;
	}

	int diff = quad->neighborCount - startingNeighborCount;
	int indexForCode = 0;
	for (int i = 0; i < quad->neighborCount; i++)
	{
		if (neighbors[i].indexInMap < 0) {
			NeighborStruct neighbor;
			neighbor.indexInMap = neighborQuads[diff-1];
			neighbor.quadCode = neighborCodes[indexForCode];
			neighbors[i] = neighbor;
			diff--;
			indexForCode++;
			if (diff == 0) { break;}
		}
	}
	if (freeMem) {
		free(neighborQuads);
	}
	free(neighborCodes);
}

__device__ void neighborsForQuadDev(NeighborStruct* neighbors, QuadStruct* quad, HashMap<int, int> *hashmap)
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