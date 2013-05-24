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
QuadStruct **QmapTexture, **QAgents;
int quadCount, QgoalCount, Qmaps, lastTextureIndex;
vector<int> QagentCount;
QuadStruct test;

extern "C" void createHashMap(QuadStruct quads[], int numberOfQuads);
extern "C" QuadStruct* quadForCode(int code);
extern "C" void neighborsForQuad(QuadStruct* quad, QuadStruct* neighbors);

extern "C" void QcomputeCostsCuda(QuadStruct* mapTexture, int numberOfQuads, int locality, int agentsNumber, QuadStruct* agents);
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
	*pred = QmapTexture[mapNumber][quad->indexInMap].prevQuadCode;
}

extern "C" EXPORT void getQuadForCode(QuadStruct* outQuad, int code)
{
	QuadStruct *quad = quadForCode(code);
	outQuad->maxx = quad->maxx;
	outQuad->minx = quad->minx;
	outQuad->maxz = quad->maxz;
	outQuad->minz = quad->minz;
	outQuad->g = quad->g;
	outQuad->costToReach = quad->costToReach;
	outQuad->inconsistent = quad->inconsistent;
	outQuad->prevQuadCode = quad->prevQuadCode;
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
	goalState->g = 0.0f;
	goalState->costToReach = cost;
	QmapTexture[mapNumber][goalState->indexInMap] = *goalState;
}

extern "C" EXPORT void insertStartQuad(int code, float cost, int agentNumber, int mapNumber) {
	QuadStruct *startState = (QuadStruct*)malloc(sizeof(QuadStruct));
	getQuadForCode(startState, code); 
	startState->g = -3.0f;
	startState->costToReach = cost;
	QmapTexture[mapNumber][startState->indexInMap] = *startState;
	QAgents[mapNumber][agentNumber] = *startState;
}

extern "C" EXPORT void generateTextureQuads(int _quadCount, int _maps, QuadStruct quads[]) {
	for (int index = 0; index < Qmaps; index++) 
	{
		free(QmapTexture[index]);
	}
	Qmaps = _maps;
	quadCount = _quadCount;
	QmapTexture = (QuadStruct**) malloc(sizeof(QuadStruct*)*Qmaps);
	size_t mapTextureSize = (quadCount*sizeof(QuadStruct));

	for (int index = 0; index < Qmaps; index++) {
		QmapTexture[index] = (QuadStruct*) malloc(mapTextureSize);
		for (int j = 0; j < quadCount; j++) {
			quads[j].indexInMap = j;
			QmapTexture[index][j] = quads[j];
		}
	}
}

extern "C" EXPORT void computeCostsMinIndexQuad(int mapNumber) {
	QcomputeCostsCuda(QmapTexture[mapNumber], quadCount, 2, QagentCount[mapNumber], QAgents[mapNumber]);
}

extern "C" EXPORT void computeCostsSubOptimalQuad(int mapNumber) {
	QcomputeCostsCuda(QmapTexture[mapNumber], quadCount, 0, QagentCount[mapNumber], QAgents[mapNumber]);
}

extern "C" EXPORT void computeCostsOptimalQuad(int mapNumber) {
	QcomputeCostsCuda(QmapTexture[mapNumber], quadCount, 1, QagentCount[mapNumber], QAgents[mapNumber]);
}

extern "C" EXPORT void insertValuesInMapQuad(int quadCode, float g, float cost, bool inconsistent, int index, int mapNumber, float minx, float maxx, float minz, float maxz) {
	QmapTexture[mapNumber][index].g = g;
	QmapTexture[mapNumber][index].costToReach = cost;
	QmapTexture[mapNumber][index].inconsistent = inconsistent;
	QmapTexture[mapNumber][index].minx = minx;
	QmapTexture[mapNumber][index].maxx = maxx;
	QmapTexture[mapNumber][index].minz = minz;
	QmapTexture[mapNumber][index].maxz = maxz;
	QmapTexture[mapNumber][index].quadCode = quadCode;
	QmapTexture[mapNumber][index].indexInMap = index;
	QmapTexture[mapNumber][index].neighborCount = 0;
}

extern "C" EXPORT void quadIn(QuadStruct quad)
{
	test = quad;
}

extern "C" EXPORT void quadOut(QuadStruct* quad)
{
	*quad = test;
}