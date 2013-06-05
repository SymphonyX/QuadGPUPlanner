#include "Structs.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <iostream>

using namespace std;

#define EXPORT __declspec(dllexport)
#define CALL __stdcall

QuadStruct **QmapTexture, **QAgents;
int quadCount, Qmaps = 0;
vector<int> QagentCount;
QuadStruct test;

extern "C" void createHashMap(QuadStruct quads[], int numberOfQuads, int size);
extern "C" int* quadForCode(int code);
extern "C" void neighborsForQuad(QuadStruct* quad, QuadStruct* neighbors);
extern "C" void cleanupDevice();

extern "C" void QcomputeCostsCuda(QuadStruct* mapTexture, int numberOfQuads, int locality, int agentsNumber, QuadStruct* agents, int goalNumber);
extern "C" void clearTextureValuesQuad(QuadStruct* mapTexture, int numberOfQuads, int goalNumber);
extern "C" EXPORT void generateTextureQuads(int _quadCount, int _maps, QuadStruct quads[]);
extern "C" void computeNeighbors(QuadStruct* texture, int numberOfQuads);

extern "C" EXPORT void computeQuadNeighborsCUDA()
{
	computeNeighbors(QmapTexture[0], quadCount);
}

extern "C" EXPORT void initQuadHashMap(QuadStruct quads[], int numberOfQuads, int numberOfMaps, int hashSize)
{
	generateTextureQuads(numberOfQuads, numberOfMaps, quads);
	createHashMap(quads, numberOfQuads, hashSize);
}

extern "C" EXPORT void updateTreeWithQuads(QuadStruct quadsInserted[], QuadStruct quadsRemoved[])
{

}

extern "C" EXPORT void getPredecessorCodeForCode(int code, int mapNumber, int* pred)
{
	int *quadIndex = quadForCode(code);
	*pred = QmapTexture[mapNumber][*quadIndex].prevQuadCode;
}

extern "C" EXPORT void getPredecessorAtIndex(int index, int mapNumber, int* pred)
{
	*pred = QmapTexture[mapNumber][index].prevQuadCode;
}

extern "C" EXPORT void getQuadForCode(QuadStruct* outQuad, int code, int mapNumber)
{
	int *quadIndex = quadForCode(code);
	QuadStruct* quad = &QmapTexture[mapNumber][*quadIndex];
	outQuad->centerx = quad->centerx;
	outQuad->centery = quad->centery;
	outQuad->g = quad->g;
	outQuad->costToReach = quad->costToReach;
	outQuad->inconsistent = quad->inconsistent;
	outQuad->prevQuadCode = quad->prevQuadCode;
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
	QmapTexture[mapNumber][goalState->indexInMap].g = 0.0f;
}

extern "C" EXPORT void insertStartQuad(int code, float cost, int agentNumber, int mapNumber) {
	QuadStruct *startState = (QuadStruct*)malloc(sizeof(QuadStruct));
	getQuadForCode(startState, code, mapNumber); 
	QAgents[mapNumber][agentNumber] = *startState;
}

extern "C" EXPORT void generateTextureQuads(int _quadCount, int _maps, QuadStruct quads[]) {
	quadCount = _quadCount;
	for (int index = 0; index < Qmaps; index++) 
	{
		free(QmapTexture[index]);
	}
	QmapTexture = (QuadStruct**)malloc(_maps*sizeof(QuadStruct*));

	size_t mapTextureSize = (quadCount*sizeof(QuadStruct));
	Qmaps = _maps;

	for (int i = 0; i < Qmaps; i++) {
		QmapTexture[i] = (QuadStruct*) malloc(mapTextureSize);
		for (int j = 0; j < quadCount; j++) {
			quads[j].indexInMap = j;
			QmapTexture[i][j] = quads[j];
		}
	}
}

extern "C" EXPORT void computeCostsMinIndexQuad(int mapNumber) {
	QcomputeCostsCuda(QmapTexture[mapNumber], quadCount, 2, QagentCount[mapNumber], QAgents[mapNumber], mapNumber);
}

extern "C" EXPORT void computeCostsSubOptimalQuad(int mapNumber) {
	QcomputeCostsCuda(QmapTexture[mapNumber], quadCount, 0, QagentCount[mapNumber], QAgents[mapNumber], mapNumber);
}

extern "C" EXPORT void computeCostsOptimalQuad(int mapNumber) {
	QcomputeCostsCuda(QmapTexture[mapNumber], quadCount, 1, QagentCount[mapNumber], QAgents[mapNumber], mapNumber);
}

extern "C" EXPORT void updateAfterGoalMovementQuad(int goalNumber)
{
	clearTextureValuesQuad(QmapTexture[goalNumber], quadCount, goalNumber);
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
	cleanupDevice();
}
