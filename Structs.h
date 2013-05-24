#ifndef STRUCTS_H
#define STRUCTS_H

struct QuadStruct {
	float minx;
	float maxx;
	float minz;
	float maxz;
	float g;
	float costToReach;
	bool inconsistent;
	int indexInMap;
	int quadCode;
	int prevQuadCode;
	int neighborCount;
};

#endif