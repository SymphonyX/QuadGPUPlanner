#include "Structs.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <iostream>

using namespace std;

#define EXPORT __declspec(dllexport)
#define CALL __stdcall

QuadStruct *QGoals;
QuadStruct *QmapTexture, **QAgents;
int quadCount, QgoalCount, Qmaps, lastTextureIndex;
vector<int> QagentCount;
QuadStruct test;

extern "C" void createHashMap(QuadStruct quads[], int numberOfQuads);
extern "C" QuadStruct* quadForCode(int code);
extern "C" void neighborsForQuad(QuadStruct* quad, QuadStruct* neighbors);
extern "C" void cleanupDevice();

extern "C" void QcomputeCostsCuda(QuadStruct* mapTexture, int numberOfQuads, int locality, int agentsNumber, QuadStruct* agents, int goalNumber);
extern "C" void clearTextureValuesQuad(QuadStruct* mapTexture, int numberOfQuads);
extern "C" EXPORT void generateTextureQuads(int _quadCount, int _maps, QuadStruct quads[]);

extern "C" EXPORT void initQuadHashMap(QuadStruct quads[], int numberOfQuads, int numberOfMaps)
{
	generateTextureQuads(numberOfQuads, numberOfMaps, quads);
	createHashMap(quads, numberOfQuads);
}

extern "C" EXPORT void getPredecessorCodeForCode(int code, int mapNumber, int* pred)
{
	QuadStruct *quad = quadForCode(code);
	*pred = QmapTexture[quad->indexInMap].prevQuadCode[mapNumber];
}

extern "C" EXPORT void getQuadForCode(QuadStruct* outQuad, int code)
{
	QuadStruct *quad = quadForCode(code);
	outQuad->maxx = quad->maxx;
	outQuad->minx = quad->minx;
	outQuad->maxz = quad->maxz;
	outQuad->minz = quad->minz;
	*(outQuad->g) = *(quad->g);
	outQuad->costToReach = quad->costToReach;
	outQuad->inconsistent = quad->inconsistent;
	*(outQuad->prevQuadCode) = *(quad->prevQuadCode);
	outQuad->quadCode = quad->quadCode;
	outQuad->neighborCount = quad->neighborCount;
	outQuad->indexInMap = quad->indexInMap;
}

extern "C" EXPORT void CALL getNeighborsForQuadCuda(QuadStruct* quad, QuadStruct neighbors[])
{
	quad->neighborCount = 0;
	QuadStruct* neighbors_ptr = (QuadStruct*)malloc(sizeof(QuadStruct)*10);
	neighborsForQuad(quad, neighbors_ptr);
	for (int i = 0; i < quad->neighborCount; i++)
	{
		neighbors[i] = neighbors_ptr[i];
	}
}


 extern "C" EXPORT void allocAgentsMemQuad(int agents, int mapNumber)
{
	if (QAgents == NULL) {
		QAgents = (QuadStruct**) malloc(sizeof(QuadStruct*)*Qmaps);
	}

	QAgents[mapNumber] = (QuadStruct*)malloc(agents*sizeof(QuadStruct));
	
	QagentCount.push_back(agents);
}

extern "C" EXPORT void allocGoalsMemQuad(int goals)
{
	free(QGoals);
	QGoals = (QuadStruct*)malloc(goals*sizeof(QuadStruct));
	QgoalCount = goals;
}

extern "C" EXPORT void insertGoalQuad(int code, float cost, int mapNumber) {
	QuadStruct *goalState = (QuadStruct*)malloc(sizeof(QuadStruct)); 
	getQuadForCode(goalState, code);
	QmapTexture[goalState->indexInMap].g[mapNumber] = 0.0f;
}

extern "C" EXPORT void insertStartQuad(int code, float cost, int agentNumber, int mapNumber) {
	QuadStruct *startState = (QuadStruct*)malloc(sizeof(QuadStruct));
	getQuadForCode(startState, code); 
	QmapTexture[startState->indexInMap].g[mapNumber] = -3.0f;
	QAgents[mapNumber][agentNumber] = *startState;
}

extern "C" EXPORT void generateTextureQuads(int _quadCount, int _maps, QuadStruct quads[]) {
	free(QmapTexture);

	quadCount = _quadCount;
	size_t mapTextureSize = (quadCount*sizeof(QuadStruct));
	QmapTexture = (QuadStruct*) malloc(mapTextureSize);

	for (int j = 0; j < quadCount; j++) {
		quads[j].indexInMap = j;
		QmapTexture[j] = quads[j];
	}
}

extern "C" EXPORT void computeCostsMinIndexQuad(int mapNumber) {
	QcomputeCostsCuda(QmapTexture, quadCount, 2, QagentCount[mapNumber], QAgents[mapNumber], mapNumber);
}

extern "C" EXPORT void computeCostsSubOptimalQuad(int mapNumber) {
	QcomputeCostsCuda(QmapTexture, quadCount, 0, QagentCount[mapNumber], QAgents[mapNumber], mapNumber);
}

extern "C" EXPORT void computeCostsOptimalQuad(int mapNumber) {
	QcomputeCostsCuda(QmapTexture, quadCount, 1, QagentCount[mapNumber], QAgents[mapNumber], mapNumber);
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
