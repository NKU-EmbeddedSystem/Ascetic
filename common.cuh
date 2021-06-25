#ifndef PTGRAPH_COMMON_CUH
#define PTGRAPH_COMMON_CUH

#include <iostream>
#include <chrono>
#include <fstream>
#include <math.h>
#include "gpu_kernels.cuh"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/sort.h>
#include <algorithm>
#include <thread>
#include "ArgumentParser.cuh"

using namespace std;
#define gpuErrorcheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

const static uint fragment_size = 4096;
//string testGraphPath = "/home/gxl/labproject/subway/uk-2007-04/output.txt";
const static string converPath = "/home/gxl/labproject/subway/uk-2007-04/uk-2007-04.bcsr";
const static string testGraphPath = "/home/gxl/dataset/friendster/friendster.bcsr";
//string testGraphPath = "/home/gxl/labproject/subway/uk-2007Restruct.bcsr";
//const static string testGraphPath = "/home/gxl/labproject/subway/friendster.bcsr";
//string testGraphPath = "/home/gxl/labproject/subway/friendsterRestruct.bcsr";
const static string testWeightGraphPath = "/home/gxl/labproject/subway/sk-2005.bwcsr";
const static string randomDataPath = "/home/gxl/labproject/subway/friendsterChange.random";
const static string prGraphPath = "/home/gxl/dataset/friendster/friendster.bcsc";
const static string ssspGraphPath = "/home/gxl/dataset/friendster/friendster.bwcsr";

const static uint DIST_INFINITY = std::numeric_limits<unsigned int>::max() - 1;
const static uint trunk_size = 1 << 24;

struct CommonPartitionInfo {
    uint startVertex;
    uint endVertex;
    uint nodePointerOffset;
    uint partitionEdgeSize;
};
struct PartEdgeListInfo {
    uint partActiveNodeNums;
    uint partEdgeNums;
    uint partStartIndex;
};

void
checkNeedTransferPartition(bool *needTransferPartition, CommonPartitionInfo *partitionInfoList, bool *isActiveNodeList,
                           int partitionNum, uint testNumNodes, uint &activeNum);

void checkNeedTransferPartitionOpt(bool *needTransferPartition, CommonPartitionInfo *partitionInfoList,
                                   bool *isActiveNodeList,
                                   int partitionNum, uint testNumNodes, uint &activeNum);


void getMaxPartitionSize(unsigned long &max_partition_size, unsigned long &totalSize, uint testNumNodes, float param,
                         int edgeSize, int nodeParamSize = 15);


void getMaxPartitionSize(unsigned long &max_partition_size, unsigned long &totalSize, uint testNumNodes, float param,
                         int edgeSize, uint edgeListSize, int nodeParamsSize = 15);

void caculatePartInfoForEdgeList(uint *overloadNodePointers, uint *overloadNodeList, uint *degree,
                                 vector<PartEdgeListInfo> &partEdgeListInfoArr, uint overloadNodeNum,
                                 uint overloadMemorySize, uint overloadEdgeNum);

static void fillDynamic(int tId,
                        int numThreads,
                        unsigned int overloadNodeBegin,
                        unsigned int numActiveNodes,
                        unsigned int *outDegree,
                        unsigned int *activeNodesPointer,
                        unsigned int *nodePointer,
                        unsigned int *activeNodes,
                        uint *edgeListOverload,
                        uint *edgeList) {
    float waitToHandleNum = numActiveNodes - overloadNodeBegin;
    float numThreadsF = numThreads;
    unsigned int chunkSize = ceil(waitToHandleNum / numThreadsF);
    unsigned int left, right;
    left = tId * chunkSize + overloadNodeBegin;
    right = min(left + chunkSize, numActiveNodes);
    unsigned int thisNode;
    unsigned int thisDegree;
    unsigned int fromHere;
    unsigned int fromThere;
    for (unsigned int i = left; i < right; i++) {
        thisNode = activeNodes[i];
        thisDegree = outDegree[thisNode];
        fromHere = activeNodesPointer[i];
        fromThere = nodePointer[thisNode];
        for (unsigned int j = 0; j < thisDegree; j++) {
            edgeListOverload[fromHere + j] = edgeList[fromThere + j];
        }
    }
}


static void writeTrunkVistInIteration(vector<vector<uint>> recordData, const string& outputPath) {
    ofstream fout(outputPath);
    for (int i = 0; i < recordData.size(); i++) {
        // output by iteration
        for (int j = 0; j < recordData[i].size(); j++) {
            fout << recordData[i][j] << "\t";
        }
        fout << endl;
    }
    fout.close();
}

static vector<uint> countDataByIteration(uint edgeListSize, uint nodeListSize, uint* nodePointers, uint* degree, int *isActive) {
    uint partSizeCursor = 0;
    uint partSize = trunk_size / sizeof(uint);
    uint partNum = edgeListSize / partSize;
    vector<uint> thisIterationVisit(partNum + 1);
    for (uint i = 0; i < nodeListSize; i++) {
        uint edgeStartIndex = nodePointers[i];
        uint edgeEndIndex = nodePointers[i] + degree[i];
        uint maxPartIndex = partSizeCursor * partSize + partSize;

        if (edgeStartIndex < maxPartIndex && edgeEndIndex < maxPartIndex) {
            if(isActive[i]) thisIterationVisit[partSizeCursor] += degree[i];
        } else if (edgeStartIndex < maxPartIndex && edgeEndIndex >= maxPartIndex) {
            if(isActive[i]) thisIterationVisit[partSizeCursor] += (maxPartIndex - edgeStartIndex);
            partSizeCursor += 1;
            if(isActive[i]) thisIterationVisit[partSizeCursor] += (edgeEndIndex - maxPartIndex);
        } else {
            partSizeCursor += 1;
            if(isActive[i]) thisIterationVisit[partSizeCursor] += degree[i];
        }
    }
    return thisIterationVisit;
}

static vector<uint> countDataByIteration(uint edgeListSize, uint nodeListSize, uint* nodePointers, uint* degree, uint *isActive) {
    uint partSizeCursor = 0;
    uint partSize = trunk_size / sizeof(uint);
    uint partNum = edgeListSize / partSize;
    vector<uint> thisIterationVisit(partNum + 1);
    for (uint i = 0; i < nodeListSize; i++) {
        uint edgeStartIndex = nodePointers[i];
        uint edgeEndIndex = nodePointers[i] + degree[i];
        uint maxPartIndex = partSizeCursor * partSize + partSize;

        if (edgeStartIndex < maxPartIndex && edgeEndIndex < maxPartIndex) {
            if(isActive[i]) thisIterationVisit[partSizeCursor] += degree[i];
        } else if (edgeStartIndex < maxPartIndex && edgeEndIndex >= maxPartIndex) {
            if(isActive[i]) thisIterationVisit[partSizeCursor] += (maxPartIndex - edgeStartIndex);
            partSizeCursor += 1;
            if(isActive[i]) thisIterationVisit[partSizeCursor] += (edgeEndIndex - maxPartIndex);
        } else {
            partSizeCursor += 1;
            if(isActive[i]) thisIterationVisit[partSizeCursor] += degree[i];
        }
    }
    return thisIterationVisit;
}

static void calculateDegree(uint nodesSize, uint* nodePointers, uint edgesSize, uint* degree) {
    for (uint i = 0; i < nodesSize - 1; i++) {
        if (nodePointers[i] > edgesSize) {
            cout << i << "   " << nodePointers[i] << endl;
            break;
        }
        degree[i] = nodePointers[i + 1] - nodePointers[i];
    }
    degree[nodesSize - 1] = edgesSize - nodePointers[nodesSize - 1];
}



#endif //PTGRAPH_COMMON_CUH