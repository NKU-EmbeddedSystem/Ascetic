#include "common.cuh"

void getMaxPartitionSize(unsigned long &max_partition_size, unsigned long &totalSize, uint testNumNodes, float param,
                         int edgeSize, int nodeParamsSize) {
    //caculate max memory
    int deviceID;
    cudaDeviceProp dev;
    cudaGetDevice(&deviceID);
    cudaGetDeviceProperties(&dev, deviceID);
    size_t totalMemory;
    size_t availMemory;
    cudaMemGetInfo(&availMemory, &totalMemory);
    long reduceMem = nodeParamsSize * sizeof(uint) * (long) testNumNodes;
    totalSize = (availMemory - reduceMem) / edgeSize;
    max_partition_size = param * totalSize;
    printf("total memory is %ld max memory is %ld, most edge size is %ld\n total edge size %ld \n multiprocessors %d \n",
           availMemory - reduceMem,
           dev.totalGlobalMem, max_partition_size, totalSize, dev.multiProcessorCount);
    if (max_partition_size > DIST_INFINITY) {
        printf("bigger than DIST_INFINITY\n");
        max_partition_size = DIST_INFINITY;
    }
    uint temp = max_partition_size % fragment_size;
    max_partition_size = max_partition_size - temp;
}


void getMaxPartitionSize(unsigned long &max_partition_size, unsigned long &totalSize, uint testNumNodes, float param,
                         int edgeSize, uint edgeListSize, int nodeParamsSize) {
    int deviceID;
    cudaDeviceProp dev{};
    cudaGetDevice(&deviceID);
    cudaGetDeviceProperties(&dev, deviceID);
    size_t totalMemory;
    size_t availMemory;
    cudaMemGetInfo(&availMemory, &totalMemory);
    long reduceMem = nodeParamsSize * sizeof(uint) * (long) testNumNodes;
    cout << "reduceMem " << reduceMem << " testNumNodes " << testNumNodes << " nodeParamsSize " << nodeParamsSize
         << endl;
    totalSize = (availMemory - reduceMem) / edgeSize;

    float adviseK = (10 - (float) edgeListSize / (float) totalSize) / 9;
    //uint dynamicDataMax = edgeListSize * edgeSize -
    /*double tempUpper = ((double) (availMemory - reduceMem) * 15 - (double)edgeListSize * edgeSize);
    double tempLower = (double) (availMemory - reduceMem) * 14;
    double adviseK = tempUpper / tempLower;*/
    cout<<"adviseK " << adviseK << endl;
    if (adviseK < 0) {
        adviseK = 0.5;
        cout<<"adviseK " << adviseK << endl;
    }
    if (adviseK > 1) {
        adviseK = 0.95;
        cout<<"adviseK " << adviseK << endl;
    }
    if (param > 0) {
        adviseK = param;
    }

    max_partition_size = adviseK * totalSize;
    cout << "availMemory " << availMemory << " totalMemory " << totalMemory << endl;
    printf("total memory is %ld totalGlobalMem is %ld, most edge size is %ld\n total edge size %ld \n multiprocessors %d adviseK %f\n",
           availMemory - reduceMem,
           dev.totalGlobalMem, max_partition_size, totalSize, dev.multiProcessorCount, adviseK);
    if (max_partition_size > DIST_INFINITY) {
        printf("bigger than DIST_INFINITY\n");
        max_partition_size = DIST_INFINITY;
    }
    uint temp = max_partition_size % fragment_size;
    max_partition_size = max_partition_size - temp;
}

void
checkNeedTransferPartition(bool *needTransferPartition, CommonPartitionInfo *partitionInfoList, bool *isActiveNodeList,
                           int partitionNum, uint testNumNodes, uint &activeNum) {
    uint tempMinNode = UINT_MAX;
    uint tempMaxNode = 0;
    for (uint j = 0; j < testNumNodes; j++) {
        if (isActiveNodeList[j]) {
            if (j < tempMinNode) {
                tempMinNode = j;
            }
            if (j > tempMaxNode) {
                tempMaxNode = j;
            }
            activeNum++;
        }
    }
    if (activeNum <= 0) {
        return;
    }
    for (int i = 0; i < partitionNum; i++) {
        needTransferPartition[i] = false;
        if (partitionInfoList[i].startVertex <= tempMaxNode && partitionInfoList[i].endVertex >= tempMinNode) {
            needTransferPartition[i] = true;
        }
    }
}

void checkNeedTransferPartitionOpt(bool *needTransferPartition, CommonPartitionInfo *partitionInfoList,
                                   bool *isActiveNodeList, int partitionNum, uint testNumNodes, uint &activeNum) {
    for (int i = 0; i < partitionNum; i++) {
        needTransferPartition[i] = false;
    }
    for (uint j = 0; j < testNumNodes; j++) {
        if (isActiveNodeList[j]) {
            for (int i = 0; i < partitionNum; i++) {
                if (partitionInfoList[i].startVertex <= j && partitionInfoList[i].endVertex >= j) {
                    needTransferPartition[i] = true;
                }
            }
            activeNum++;
        }
    }
}

void caculatePartInfoForEdgeList(uint *overloadNodePointers, uint *overloadNodeList, uint *degree,
                                 vector<PartEdgeListInfo> &partEdgeListInfoArr, uint overloadNodeNum,
                                 uint overloadMemorySize, uint overloadEdgeNum) {
    partEdgeListInfoArr.clear();
    if (overloadMemorySize < overloadEdgeNum) {
        uint left = 0;
        uint right = overloadNodeNum - 1;
        while ((overloadNodePointers[right] + degree[overloadNodeList[right]] - overloadNodePointers[left]) >
               overloadMemorySize) {
            uint start = left;
            uint end = right;
            uint mid;
            while (start <= end) {
                mid = (start + end) / 2;
                uint headDistance = overloadNodePointers[mid] - overloadNodePointers[left];
                uint tailDistance =
                        overloadNodePointers[mid] + degree[overloadNodeList[mid]] - overloadNodePointers[left];
                if (headDistance <= overloadMemorySize && tailDistance > overloadMemorySize) {
                    break;
                } else if (tailDistance <= overloadMemorySize) {
                    start = mid + 1;
                } else if (headDistance > overloadMemorySize) {
                    end = mid - 1;
                }
            }
            PartEdgeListInfo info;
            info.partActiveNodeNums = mid - left;
            info.partEdgeNums = overloadNodePointers[mid] - overloadNodePointers[left];
            info.partStartIndex = left;
            partEdgeListInfoArr.push_back(info);
            left = mid;
        }
        PartEdgeListInfo info;
        info.partActiveNodeNums = right - left + 1;
        info.partEdgeNums = overloadNodePointers[right] + degree[overloadNodeList[right]] - overloadNodePointers[left];
        info.partStartIndex = left;
        partEdgeListInfoArr.push_back(info);
    } else {
        PartEdgeListInfo info;
        info.partActiveNodeNums = overloadNodeNum;
        info.partEdgeNums = overloadEdgeNum;
        info.partStartIndex = 0;
        partEdgeListInfoArr.push_back(info);
    }
}

