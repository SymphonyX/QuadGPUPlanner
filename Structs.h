#ifndef STRUCTS_H
#define STRUCTS_H

struct QuadStruct {
	float minx;
	float maxx;
	float minz;
	float maxz;
	float g[10];
	float costToReach;
	bool inconsistent;
	int indexInMap;
	int quadCode;
	int prevQuadCode[10];
	int neighborCount;
};

#endif