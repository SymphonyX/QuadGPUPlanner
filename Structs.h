#ifndef STRUCTS_H
#define STRUCTS_H

struct QuadStruct {
	float centerx;
	float centery;
	float g;
	float costToReach;
	bool inconsistent;
	int indexInMap;
	int quadCode;
	int prevQuadCode;
	int neighborCount;
};

#endif