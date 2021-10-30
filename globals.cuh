#ifndef GLOBALS_CUH
#define GLOBALS_CUH
enum ALG_TYPE {
    BFS, SSSP, CC, PR
};
typedef uint SIZE_TYPE;
//typedef unsigned long long EDGE_POINTER_TYPE;
typedef ull EDGE_POINTER_TYPE;

struct EdgeWithWeight {
    uint toNode;
    uint weight;
};
struct FragmentData {
    uint startVertex = UINT_MAX - 1;
    uint vertexNum = 0;
    bool isIn = false;
    bool isVisit = false;
    FragmentData() {
        startVertex = UINT_MAX - 1;
        vertexNum = 0;
        isIn = false;
        isVisit = false;
    }
};

#endif