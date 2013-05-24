#include <math.h>
#include "Structs.h"
#include <vector>
#include <stdlib.h>
#include <stdio.h>
#include "hash_map_template.h"
 
#define	STARTING_VALUE -1
#define OBSTACLE_VALUE -2
#define GOAL_VALUE -3
#define BLOCK_SIZE 256
//Defines Texture and its methods

using namespace CUDASTL;

HashMap<int, QuadStruct>* hashmap;
__device__ void neighborsForQuadDev(int* neighbors, QuadStruct* quad, HashMap<int, QuadStruct> *hashmap);

__device__ bool equals(float x1, float x2)
{
	if (fabs(x1 - x2) < .01) {
		return true;
	}
	return false;
}

__device__ float distance(QuadStruct *from, QuadStruct *to) {
	float fromx = (from->maxx+from->minx)/2;
	float fromy = (from->maxz+from->minz)/2;
	float tox = (to->maxx+to->minx)/2;
	float toy = (to->maxz+to->minz)/2;
	
	if (equals(fromx,tox)) {
		return fabs(toy - fromy);
	}
	else if(equals(fromy,toy)) {
		return fabs(tox - fromx);
	}
	else {
		return sqrt(fabs(tox - fromx)*fabs(tox - fromx) +
			fabs(toy - fromy)*fabs(toy - fromy));
	}
}

__device__ int stateNeedsUpdate(QuadStruct* state) {
	return state->g == STARTING_VALUE || state->g == GOAL_VALUE;
}

/*__device__ int stateIsObstacle(StateStruct* state) {
	return state->costToReach > 10.0f;
}*/

__device__ int QisGoalState(QuadStruct* state) {
	return state->g == 0.0f;
}

//Kernel function for planner

__global__ void QcomputeCostsKernel(QuadStruct *current_texture, QuadStruct *texture_copy, HashMap<int, QuadStruct> *hashmap, int numberOfQuads, int *check, int *locality, int agentCount, QuadStruct* agents, float maxCost, bool allAgentsReached) {
	int id = threadIdx.x;


	if (id < numberOfQuads) {
		QuadStruct *quad = &current_texture[id];

		//if(!stateIsObstacle(state) && !isGoalState(state)) {
			//if the state is an obstacle, do not compute neighbors
		if(!QisGoalState(quad)) {
			int neighborIndexes[] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
			neighborsForQuadDev(neighborIndexes, quad, hashmap);

			int predecesorIndex;
			for (int i = 0; i < quad->neighborCount; ++i) {
				QuadStruct *neighbor = &texture_copy[neighborIndexes[i]]; //Needs to find a quad in the ro map

				//if (stateIsObstacle(neighbor)) //if neighbor is an obstacle, do not use it as a possible neighbor
					//continue;
				float newg = neighbor->g + distance(neighbor, quad) + quad->costToReach;
				if ((newg < quad->g || stateNeedsUpdate(quad)) && !stateNeedsUpdate(neighbor)) {
					predecesorIndex = neighbor->indexInMap;
					quad->prevQuadCode = neighbor->quadCode;
					quad->g = newg;
					if (*locality == 1) {
						*check = 0;
					} else if (*locality == 2) {
						if (quad->g < maxCost || !allAgentsReached) {
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

float agentsMaxCost(QuadStruct* texture, int agentCount, QuadStruct* agents) {
	float maxCost = -10000.0f;
	for (int i = 0; i < agentCount; i++)  {
		QuadStruct agent = texture[agents[i].indexInMap];
		if (agent.g > maxCost) {
			maxCost = agent.g;
		}
	}
	return maxCost;
}

bool agentsReached(QuadStruct* texture, int agentCount, QuadStruct* agents) {
	for (int i = 0; i < agentCount; i++) {
		QuadStruct agent = texture[agents[i].indexInMap];
		if (agent.g < 0.0f) {
			return false;
		}
	}
	return true;
}


extern "C" int QcomputeCostsCuda(QuadStruct* texture, int numberOfQuads, int locality, int agentCount, QuadStruct* agents) {
	int *locality_dev, *consistencyCheck, *consistencyCheck_dev, *flag, *flag_dev;
	
	int blockLength = sqrt((double)BLOCK_SIZE); 
	int gridLength = ceil((double)numberOfQuads/(double)blockLength);
	
	dim3 blocks(1, 1, 1);
	dim3 threads(numberOfQuads, 1, 1);
	
	QuadStruct *texture_device, *texture_device_copy;
	cudaMalloc((void**)&texture_device, (numberOfQuads)*sizeof(QuadStruct));
	cudaMalloc((void**)&texture_device_copy, (numberOfQuads)*sizeof(QuadStruct));
	//make a two copies of the initial map
	cudaMemcpy(texture_device, texture, (numberOfQuads)*sizeof(QuadStruct), cudaMemcpyHostToDevice);
	cudaMemcpy(texture_device_copy, texture, (numberOfQuads)*sizeof(QuadStruct), cudaMemcpyHostToDevice);

	
	QuadStruct *agents_device;
	cudaMalloc((void**)&agents_device, (agentCount)*sizeof(QuadStruct));
	cudaMemcpy(agents_device, agents, (agentCount)*sizeof(QuadStruct), cudaMemcpyHostToDevice);

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

		bool allAgentsReached = agentsReached(texture, agentCount, agents);
		float maxCost;
		if (allAgentsReached) {
			maxCost = agentsMaxCost(texture, agentCount, agents);
		}
		QcomputeCostsKernel<<<blocks, threads>>>(texture_device, texture_device_copy, hashmap, numberOfQuads, consistencyCheck_dev, locality_dev, agentCount, agents_device, maxCost, allAgentsReached);
		
		checkForInconsistency<<<blocks, threads>>>(texture_device, numberOfQuads, flag_dev);
		
		QuadStruct* temp = texture_device;
		texture_device = texture_device_copy;
		texture_device_copy = temp;
		iterations++;
		
		cudaMemcpy(texture, texture_device, (numberOfQuads)*sizeof(QuadStruct), cudaMemcpyDeviceToHost);
		
		cudaMemcpy(consistencyCheck, consistencyCheck_dev, sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(flag, flag_dev, sizeof(int), cudaMemcpyDeviceToHost);
	} while(*consistencyCheck == 0 || *flag == 1);

	
	cudaFree(texture_device); cudaFree(texture_device_copy); cudaFree(agents_device);

	printf("Result was: %i\n\n", *consistencyCheck);
	printf("Number of iterations: %i\n\n", iterations);

	return 1;
}

__global__ void clearTextureValuesKernel(QuadStruct* texture, int numberOfQuads) {
	int id = get_thread_id();

	if (id < numberOfQuads) {
		QuadStruct* state = &texture[id];
		state->g = STARTING_VALUE;
		state->prevQuadCode = 0;
		state->inconsistent = false;
		state->neighborCount = 0;
	}

}

extern "C" void QclearTextureValues(QuadStruct* texture, int numberOfQuads) {
	QuadStruct* texture_dev;
	int blockLength = sqrt((double)BLOCK_SIZE); 
	int gridLength = ceil((double)numberOfQuads/(double)blockLength);
	dim3 blocks(gridLength, gridLength, 1);
	dim3 threads(blockLength, blockLength, 1);

	cudaMalloc((void**)&texture_dev, (numberOfQuads*sizeof(QuadStruct)));
	cudaMemcpy(texture_dev, texture, numberOfQuads*sizeof(QuadStruct), cudaMemcpyHostToDevice);
	clearTextureValuesKernel<<<blocks, threads>>> (texture_dev, numberOfQuads);
	cudaMemcpy(texture, texture_dev, (numberOfQuads)*sizeof(QuadStruct), cudaMemcpyDeviceToHost);

	cudaFree(texture_dev);
}

__global__ void populateHashMap(HashMap<int, QuadStruct> *hash,  QuadStruct *quads, int numberOfQuads)
{
	int i = get_thread_id();
	if (i < numberOfQuads) {
		(*hash)[quads[i].quadCode] = quads[i];
	}
}

extern "C" void createHashMap(QuadStruct quads[], int numberOfQuads)
{
	hashmap = CreateHashMap<int,QuadStruct, HashFunc<int> >(1, numberOfQuads);
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

__global__ void retrieveQuadStruct(QuadStruct* quad, int code, HashMap<int, QuadStruct> *hashmap)
{
	int i = get_thread_id();
	if (i == 1) {
		QuadStruct q = *((*hashmap).valueForKey(code));
		*quad = q; 
	}
}

extern "C" QuadStruct *quadForCode(int code)
{
	QuadStruct *q_dev, *q;
	q = (QuadStruct*)malloc(sizeof(QuadStruct));
	cudaMalloc((void**)&q_dev, sizeof(QuadStruct));
	
	retrieveQuadStruct<<<1, 512>>> (q_dev, code, hashmap);
	cudaMemcpy(q, q_dev, sizeof(QuadStruct), cudaMemcpyDeviceToHost);
	cudaFree(q_dev);
	return q;
}


__global__ void getNeighborsKernel(QuadStruct* neighbors, QuadStruct* quad, HashMap<int, QuadStruct> *hashmap)
{
//	neighborsForQuadDev(neighbors, quad, hashmap);
}

extern "C" void neighborsForQuad(QuadStruct* quad, QuadStruct* neighbors)
{
	QuadStruct *q_dev;
	QuadStruct *neighbors_dev;

	cudaMalloc((void**)&q_dev, sizeof(QuadStruct));
	cudaMemcpy(q_dev, quad, sizeof(QuadStruct), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&neighbors_dev, sizeof(QuadStruct)*10);
	
	getNeighborsKernel<<<1, 1>>> (neighbors_dev, q_dev, hashmap);
	cudaMemcpy(quad, q_dev, sizeof(QuadStruct), cudaMemcpyDeviceToHost);
	cudaMemcpy(neighbors, neighbors_dev, sizeof(QuadStruct)*10, cudaMemcpyDeviceToHost);
	cudaFree(q_dev);
	cudaFree(neighbors_dev);
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

__device__ QuadStruct* greaterNeighbor(QuadStruct quad, int neighborQuadCode, HashMap<int, QuadStruct> *hashmap)
{
	int parsedCode = neighborQuadCode;

	QuadStruct* neighbor = NULL;
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

__device__ void populateSmallerNeighbors(QuadStruct* quad, QuadStruct* neighbors, int quadCode, int addCode1, int addCode2, HashMap<int, QuadStruct> *hashmap, int indexOffset)
{
	QuadStruct* neighbor = (*hashmap).valueForKey(quadCode);

	if (neighbor != NULL) {
		neighbors[indexOffset] = *neighbor;
		quad->neighborCount += 1;
	} else {
		int *codes = (int*)malloc(sizeof(int)*(16));
		codes[0] = (quadCode*10) + addCode1;
		codes[1] = (quadCode*10) + addCode2;
		int lastIndex = 2;
		for (int i = 0; i < lastIndex; i++) {
			int code = codes[i];
			if (code/1000000 > 0) { continue; }

			QuadStruct* subNeighbor = (*hashmap).valueForKey(code);
			if (subNeighbor != NULL) {
				neighbors[indexOffset++] = *subNeighbor;
				quad->neighborCount += 1;
			} else {
				codes[lastIndex] = (code*10) + addCode1;
				codes[lastIndex+1] = (code*10) + addCode2;
				lastIndex += 2;
			}
		}
	}
}

__device__ QuadStruct* smallerNeighbors(QuadStruct* quad, int neighborQuadCode, int addCode1, int addCode2, HashMap<int, QuadStruct> *hashmap)
{
	QuadStruct* neighbors = (QuadStruct*)malloc(sizeof(QuadStruct)*16);

	int neighborCode1 = (neighborQuadCode*10) + addCode1;
	int neighborCode2 = (neighborQuadCode*10) + addCode2;

	int currentNeighbors = quad->neighborCount;
	int indexOffset = 0;
	populateSmallerNeighbors(quad, neighbors, neighborCode1, addCode1, addCode2, hashmap, indexOffset);

	indexOffset = quad->neighborCount - currentNeighbors;
	populateSmallerNeighbors(quad, neighbors, neighborCode2, addCode1, addCode2, hashmap, indexOffset);

	return neighbors;
}

__device__ void retrieveNeighbors(int* neighbors, QuadStruct* quad, int max_i, HashMap<int, QuadStruct> *hashmap, int codeDiff, int addCode1, int addCode2)
{
	int neighborCode = constructNeighborQuadCode(quad, max_i, codeDiff);
	QuadStruct* neighborQuads = (*hashmap).valueForKey(neighborCode);
	int startingNeighborCount = quad->neighborCount;
	if (neighborQuads == NULL) {
		neighborQuads = greaterNeighbor(*quad, neighborCode, hashmap);
		if (neighborQuads == NULL) {
			neighborQuads = smallerNeighbors(quad, neighborCode, addCode1, addCode2, hashmap);
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
			neighbors[i] = neighborQuads[diff-1].indexInMap;
			diff--;
			if (diff == 0) { break;}
		}
	}
}

__device__ void neighborsForQuadDev(int* neighbors, QuadStruct* quad, HashMap<int, QuadStruct> *hashmap)
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