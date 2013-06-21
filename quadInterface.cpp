#include "Structs.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <iostream>

using namespace std;

#define EXPORT __declspec(dllexport)
#define CALL __stdcall

#define INVALID_QUAD -5.0f

QuadStruct *QmapTexture, **QAgents;
PlanStruct **PmapTexture;
int quadCount, Qmaps = 0;
vector<int> QagentCount;
vector<int> availableMemoryIndexes;
QuadStruct test;

extern "C" void createHashMap(QuadStruct quads[], int numberOfQuads, int size);
extern "C" int* quadForCode(int code);
extern "C" int* invalidateQuadsInHash(QuadStruct quadsRemoved[], int count);
extern "C" int* updateQuadsInHash(QuadStruct updateQuads[], int count);
extern "C" void insertNewQuadsInHash(QuadStruct quadsInserted[], int count);
extern "C" void propagateUpdateAfterObstacleMovement(PlanStruct* texture, int numberOfQuads);
extern "C" void setQuadMap(QuadStruct* texture, int numberOfQuads);
extern "C" void getNeighborCodesForIndex(int index, int* indexes, int quadCount);


extern "C" void cleanupDevice();

extern "C" void QcomputeCostsCuda(PlanStruct* map, int numberOfQuads, int locality, int agentsNumber, QuadStruct* agents, int goalNumber);
extern "C" void clearTextureValuesQuad(PlanStruct* PmapTexture, int numberOfQuads, int goalNumber);
extern "C" EXPORT void generateTextureQuads(int _quadCount, int _maps, QuadStruct quads[]);
extern "C" void computeNeighbors(QuadStruct* texture, int numberOfQuads);
extern "C" void updateNeighborsToQuads(int* indexes, int size, int totalNumberOfQuads, PlanStruct** texture, int numberOfGoals, int* updateIndexes, int updateCount);
extern "C" void repairNeighbors(QuadStruct* texture, QuadStruct quadsIn[], int numberIn, int* indexesRemoved, int countFree, int quadCount);

extern "C" EXPORT void computeQuadNeighborsCUDA()
{
	computeNeighbors(QmapTexture, quadCount);
}

extern "C" EXPORT void initQuadHashMap(QuadStruct quads[], int numberOfQuads, int numberOfMaps, int hashSize)
{  
	generateTextureQuads(numberOfQuads, numberOfMaps, quads);
	createHashMap(quads, numberOfQuads, hashSize);
}


extern "C" EXPORT void updateTreeWithQuads(QuadStruct quadsInserted[], int countInserted, QuadStruct quadsRemoved[], int countRemoved, QuadStruct updateQuads[], int updateCount)
{
	int* updateIndexes = updateQuadsInHash(updateQuads, updateCount);
	for (int i = 0; i < updateCount; i++) {
		QmapTexture[updateIndexes[i]].costToReach = updateQuads[i].costToReach;
		for (int j = 0; j < Qmaps; j++) {
			PmapTexture[j][updateIndexes[i]].g = updateQuads[i].g;
			PmapTexture[j][updateIndexes[i]].prevQuadCode = updateQuads[i].prevQuadCode;
		}
	}

	int* freeIndexes = invalidateQuadsInHash(quadsRemoved, countRemoved); 
	updateNeighborsToQuads(freeIndexes, countRemoved, quadCount, PmapTexture, Qmaps, updateIndexes, updateCount);

	for (int i = 0; i < countRemoved; i++) {
		for (int j = 0; j < Qmaps; j++) {
			PmapTexture[j][freeIndexes[i]].g = INVALID_QUAD;
		}
		QmapTexture[freeIndexes[i]].g = INVALID_QUAD;
		availableMemoryIndexes.push_back(freeIndexes[i]);
	}


	for (int i = 0; i < countInserted; i++) {
	   	int index;
		if (availableMemoryIndexes.size() > 0) {
			bool indexFound = false;
			for (int j = 0; j < availableMemoryIndexes.size(); j++) {
				if (QmapTexture[availableMemoryIndexes.at(j)].quadCode == quadsInserted[i].quadCode)
				{
					index = availableMemoryIndexes.at(j);
					availableMemoryIndexes.erase(availableMemoryIndexes.begin()+j);
					indexFound = true;
					break;
				}
			}
			if (!indexFound) {
				index = availableMemoryIndexes.back();
				availableMemoryIndexes.pop_back();
			}
		} else {
			index = quadCount++;
			QmapTexture = (QuadStruct*) realloc(QmapTexture, sizeof(QuadStruct)*quadCount);
			for (int i = 0; i < Qmaps; i++) {
				PmapTexture[i] = (PlanStruct*) realloc(PmapTexture[i], sizeof(PlanStruct)*quadCount);
			}
		}

		quadsInserted[i].indexInMap = index;
		QmapTexture[index] = quadsInserted[i];
		for (int j = 0; j < Qmaps; j++) {
			PmapTexture[j][index].g = quadsInserted[i].g;
			PmapTexture[j][index].prevQuadCode = quadsInserted[i].prevQuadCode;
		}
	}
	
	insertNewQuadsInHash(quadsInserted, countInserted);
	setQuadMap(QmapTexture, quadCount);

	//computeQuadNeighborsCUDA();
	repairNeighbors(QmapTexture, quadsInserted, countInserted, freeIndexes, countRemoved, quadCount);
	for (int i = 0; i < Qmaps; i++) {
		propagateUpdateAfterObstacleMovement(PmapTexture[i], quadCount);
	}
	free(freeIndexes);
}

extern "C" EXPORT void getPredecessorCodeForCode(int code, int mapNumber, int* pred)
{
	int *quadIndex = quadForCode(code);
	*pred = PmapTexture[mapNumber][*quadIndex].prevQuadCode;
}

extern "C" EXPORT void getPredecessorForAgent(int agentNumber, int mapNumber, int* pred)
{
	QuadStruct agent = QAgents[mapNumber][agentNumber];
	*pred = PmapTexture[mapNumber][agent.indexInMap].prevQuadCode;
}

extern "C" EXPORT void getPredecessorAtIndex(int index, int mapNumber, int* pred)
{
	*pred = PmapTexture[mapNumber][index].prevQuadCode;
}

extern "C" EXPORT void getQuadForCode(QuadStruct* outQuad, int code, int mapNumber)
{
	int *quadIndex = quadForCode(code);
	QuadStruct* quad = &QmapTexture[*quadIndex];
	outQuad->centerx = quad->centerx;
	outQuad->centery = quad->centery;
	outQuad->g = PmapTexture[mapNumber][*quadIndex].g;
	outQuad->costToReach = quad->costToReach;
	outQuad->inconsistent = quad->inconsistent;
	outQuad->prevQuadCode = PmapTexture[mapNumber][*quadIndex].prevQuadCode;
	outQuad->quadCode = quad->quadCode;
	outQuad->neighborCount = quad->neighborCount;
	outQuad->indexInMap = quad->indexInMap;
}

 extern "C" EXPORT void allocAgentsMemQuad(int agents, int mapNumber)
{
	if (QAgents == NULL) {
		QAgents = (QuadStruct**) malloc(sizeof(QuadStruct*)*Qmaps);
	}

	QAgents[mapNumber] = (QuadStruct*)malloc(agents*sizeof(QuadStruct));
	
	QagentCount.push_back(agents);
}

extern "C" EXPORT void insertGoalQuad(int code, float cost, int mapNumber) {
	QuadStruct *goalState = (QuadStruct*)malloc(sizeof(QuadStruct)); 
	getQuadForCode(goalState, code, mapNumber);
	PmapTexture[mapNumber][goalState->indexInMap].g = 0.0f;
}

extern "C" EXPORT void insertStartQuad(int code, float cost, int agentNumber, int mapNumber) {
	QuadStruct *startState = (QuadStruct*)malloc(sizeof(QuadStruct));
	getQuadForCode(startState, code, mapNumber); 
	QAgents[mapNumber][agentNumber] = *startState;
}

extern "C" EXPORT void generateTextureQuads(int _quadCount, int _maps, QuadStruct quads[]) {
	quadCount = _quadCount;
	if (QmapTexture != NULL) free(QmapTexture);
	for (int index = 0; index < Qmaps; index++) 
	{
		free(PmapTexture[index]);
	}
	QmapTexture = (QuadStruct*)malloc(quadCount*sizeof(QuadStruct));
	PmapTexture = (PlanStruct**)malloc(_maps*sizeof(PlanStruct*));

	size_t mapTextureSize = (quadCount*sizeof(PlanStruct));
	Qmaps = _maps;

	for (int i = 0; i < Qmaps; i++) {
		PmapTexture[i] = (PlanStruct*) malloc(mapTextureSize);
		for (int j = 0; j < quadCount; j++) {
			quads[j].indexInMap = j;
			QmapTexture[j] = quads[j];
			PmapTexture[i][j].g = quads[j].g;
			PmapTexture[i][j].prevQuadCode = quads[j].prevQuadCode;
		}
	}
	setQuadMap(QmapTexture, quadCount);
}

extern "C" EXPORT void computeCostsMinIndexQuad(int mapNumber) {
	QcomputeCostsCuda(PmapTexture[mapNumber], quadCount, 2, QagentCount[mapNumber], QAgents[mapNumber], mapNumber);
}

extern "C" EXPORT void computeCostsSubOptimalQuad(int mapNumber) {
	QcomputeCostsCuda(PmapTexture[mapNumber], quadCount, 0, QagentCount[mapNumber], QAgents[mapNumber], mapNumber);
}

extern "C" EXPORT void computeCostsOptimalQuad(int mapNumber) {
	QcomputeCostsCuda(PmapTexture[mapNumber], quadCount, 1, QagentCount[mapNumber], QAgents[mapNumber], mapNumber);
}

extern "C" EXPORT void updateAfterGoalMovementQuad(int goalNumber)
{
	clearTextureValuesQuad(PmapTexture[goalNumber], quadCount, goalNumber);
}

extern "C" EXPORT void quadIn(QuadStruct quad)
{
	test = quad;
}

extern "C" EXPORT void quadOut(QuadStruct* quad)
{
	*quad = test;
}

extern "C" EXPORT void cleanup()
{
	availableMemoryIndexes.clear();
	cleanupDevice();
}

extern "C" EXPORT void CALL debugGvalues(QuadStruct quads[], int mapNumber)
{
	for (int i = 0; i < quadCount; i++) {
		quads[i].g = PmapTexture[mapNumber][i].g;
	}
}



extern "C" EXPORT void CALL getNeighborCodesForCodeAtIndex(int code, int* neighborCode, int neighborIndex)
{
	int* index = quadForCode(code);
	int* indexes = (int*)malloc(sizeof(int)*20);
	getNeighborCodesForIndex(*index, indexes, quadCount);

	if (indexes[neighborIndex] >= 0) {
		*neighborCode = QmapTexture[indexes[neighborIndex]].quadCode;
	} else {
		*neighborCode = 0;
	}
	
}
