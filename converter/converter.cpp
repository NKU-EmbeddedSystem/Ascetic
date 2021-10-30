#include "./globals.hpp"
#include <cstdlib>
#include <ctime>

bool IsWeightedFormat(string format)
{
	if((format == "bwcsr")	||
		(format == "wcsr")	||
		(format == "wel"))
			return true;
	return false;
}

string GetFileExtension(string fileName)
{
    if(fileName.find_last_of(".") != string::npos)
        return fileName.substr(fileName.find_last_of(".")+1);
    return "";
}


void convertTxtToByte(string input) {
	clock_t startTime, endTime;
	startTime = clock();
	ifstream infile;
    infile.open(input);
    std::ofstream outfile(input.substr(0, input.length()-3)+"toB", std::ofstream::binary);
    stringstream ss;
    uint max = 0;
    string line;
    ull edgeCounter = 0;
    vector<Edge> edges;
    Edge newEdge;
    ull testIndex = 0;
    while(getline( infile, line ))
    {
    	testIndex++;
        ss.str("");
        ss.clear();
        ss << line;
        ss >> newEdge.source;
        ss >> newEdge.end;
        if(newEdge.source==1)
        cout << newEdge.source << "  " << newEdge.end << endl;
        edges.push_back(newEdge);
        edgeCounter++;
        if(max < newEdge.source)
            max = newEdge.source;
        if(max < newEdge.end)
            max = newEdge.end;
    	if (testIndex % 1000000000 == 1)
    	{
    		int billionLines = testIndex / 1000000000;
    		cout << billionLines << " billion lines " << endl;
    		if (billionLines % 5 == 1)
    		{
    			outfile.write((char*)edges.data(), sizeof(Edge) * edges.size());
    			edges.clear();
    			cout << "clear edges = " << edges.size() << endl;
    		}
    	}

    }
	if (edges.size() > 0)
	{
		outfile.write((char*)edges.data(), sizeof(Edge) * edges.size());
		edges.clear();
	}
	cout << "write " << testIndex << " lines" << endl;
	cout << "max node " << max << endl;
    outfile.close();
    infile.close();
    endTime = clock();
    cout << setprecision(9)  << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s | " << "S0:Prepare [edges] OK" << endl;
}

void convertByteToBCSR(string input) {
	cout << "convertByteToBCSR" << endl;
	clock_t startTime, endTime;
	startTime = clock();
	ull totalSize = 10000000000;
	uint partSize = 1000000000;
	uint vertexSize = 100000000;
	ull* nodePointers = new ull[vertexSize];
	ull* nodePointersAnkor = new ull[vertexSize];
	ull* degree = new ull[vertexSize];
	OutEdge* edges = new OutEdge[totalSize];
	Edge *byteEdges = new Edge[partSize];
	ull counter = 0;
	for(uint i = 0; i < vertexSize; i++)
	 	degree[i] = 0;

	cout << "calculate degree " << endl;
	ifstream infile(input, ios::in | ios::binary);
	infile.seekg(0, std::ios::end);
	ull size = infile.tellg();
	totalSize = size / sizeof(Edge);
	cout << "total edge Size " << totalSize << endl;
	infile.clear();
	infile.seekg(0);
	uint maxNode = 0;
	while(counter < totalSize) {
		cout << counter << " to " << counter + partSize << endl;
		counter += partSize;
		infile.read((char *) byteEdges, sizeof(Edge) * partSize);
		for (uint i = 0; i < partSize; ++i)
		{
			degree[byteEdges[i].source]++;
			if (maxNode < byteEdges[i].source)
			{
				maxNode = byteEdges[i].source;
			}
		}
	}
	//cout<<"degree[1]: "<<degree[1]<<endl;
	cout << "max node is " << maxNode << endl;
	vertexSize = maxNode + 1;
	uint tempPointer = 0;
	for(uint i=0; i<vertexSize; i++)
	{
		nodePointers[i] = tempPointer;
		tempPointer += degree[i];
	}
	for(int i=0;i<10;i++)
	{
		cout<<"dgree: "<<degree[i]<<" nodePointer: "<<nodePointers[i]<<endl;
	}
	infile.clear();
	infile.seekg(0);

	cout << "calculate edges " << endl;
	for (uint i = 0; i < vertexSize; ++i)
	{
		nodePointersAnkor[i] = 0;
	}
	counter = 0;
	while(counter < totalSize) {
		cout << counter << " to " << counter + partSize << endl;
		counter += partSize;
		infile.read((char *) byteEdges, sizeof(Edge) * partSize);
		for (uint i = 0; i < partSize; ++i)
		{
			ull location = nodePointers[byteEdges[i].source] + nodePointersAnkor[byteEdges[i].source];
	 		edges[location].end = byteEdges[i].end;
	 		nodePointersAnkor[byteEdges[i].source]++;
		}
	}
	//delete [] nodePointersAnkor;

	std::ofstream outfile(input.substr(0, input.length()-3)+"bcsr", std::ofstream::binary);
	outfile.write((char*)&vertexSize, sizeof(unsigned int));
	outfile.write((char*)&totalSize, sizeof(ull));
	outfile.write ((char*)nodePointers, sizeof(ull)*vertexSize);
	outfile.write ((char*)edges, sizeof(OutEdge)*totalSize);
	outfile.close();
	cout << "ull size is " << sizeof(ull) << endl;
	endTime = clock();
    cout << setprecision(9)  << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s | " << "S0:Prepare [edges] OK" << endl;
}

void convertByteToBCSC(string input) {
	cout << "convertByteToBCSC" << endl;
	clock_t startTime, endTime;
	startTime = clock();
	ull totalSize = 10000000000;
	uint partSize = 1000000000;
	uint vertexSize = 100000000;
	ull* nodePointers = new ull[vertexSize];
	ull* nodePointersAnkor = new ull[vertexSize];
	uint* degree = new uint[vertexSize];
	uint *inDegree = new uint[vertexSize];
	OutEdge* edges = new OutEdge[totalSize];
	Edge *byteEdges = new Edge[partSize];
	ull counter = 0;
	for(uint i = 0; i < vertexSize; i++) {
	 	degree[i] = 0;
	 	inDegree[i] = 0;
	}

	cout << "calculate degree " << endl;
	ifstream infile(input, ios::in | ios::binary);
	infile.seekg(0, std::ios::end);
	ull size = infile.tellg();
	totalSize = size / sizeof(Edge);
	cout << "total edge Size " << totalSize << endl;
	infile.clear();
	infile.seekg(0);
	uint maxNode = 0;
	while(counter < totalSize) {
		cout << counter << " to " << counter + partSize << endl;
		counter += partSize;
		infile.read((char *) byteEdges, sizeof(Edge) * partSize);
		for (uint i = 0; i < partSize; ++i)
		{
			degree[byteEdges[i].source]++;
			inDegree[byteEdges[i].end]++;
			if (maxNode < byteEdges[i].source)
			{
				maxNode = byteEdges[i].source;
			}
		}
	}
	cout << "max node is " << maxNode << endl;
	vertexSize = maxNode + 1;
	ull tempPointer = 0;
	for(uint i=0; i<vertexSize; i++)
	{
		nodePointers[i] = tempPointer;
		tempPointer += inDegree[i];
	}
	cout << "finish calculate nodePointers " << endl;
	// for (int i = 0; i < 100; ++i)
	// {
	// 	cout << i << " : " << nodePointers[i] << endl;
	// }

	// for (int i = 90000000; i < 90000000 + 100; ++i)
	// {
	// 	cout << i << " : " << nodePointers[i] << endl;
	// }

	infile.clear();
	infile.seekg(0);

	cout << "calculate edges " << endl;
	for (uint i = 0; i < vertexSize; ++i)
	{
		nodePointersAnkor[i] = 0;
	}
	counter = 0;
	while(counter < totalSize) {
		cout << counter << " to " << counter + partSize << endl;
		counter += partSize;
		infile.read((char *) byteEdges, sizeof(Edge) * partSize);
		for (uint i = 0; i < partSize; ++i)
		{
			ull location = nodePointers[byteEdges[i].source] + nodePointersAnkor[byteEdges[i].source];
	 		edges[location].end = byteEdges[i].end;
	 		nodePointersAnkor[byteEdges[i].source]++;
		}
	}
	//delete [] nodePointersAnkor;

	std::ofstream outfile(input.substr(0, input.length()-3)+"bcsc", std::ofstream::binary);
	outfile.write((char*)&vertexSize, sizeof(unsigned int));
	outfile.write((char*)&totalSize, sizeof(ull));
	outfile.write ((char*)degree, sizeof(uint)*vertexSize);
	outfile.write ((char*)nodePointers, sizeof(ull)*vertexSize);
	outfile.write ((char*)edges, sizeof(OutEdge)*totalSize);
	outfile.close();
	endTime = clock();
    cout << setprecision(9)  << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s | " << "S0:Prepare [edges] OK" << endl;
}

void convertByteToBWCSR(string input) {
	cout << "convertByteToBWCSR" << endl;
	srand(0);
	clock_t startTime, endTime;
	startTime = clock();
	ull totalSize = 6000000000;
	uint partSize = 1000000000;
	uint vertexSize = 60000000;
	ull* nodePointers = new ull[vertexSize];
	ull* nodePointersAnkor = new ull[vertexSize];
	ull* degree = new ull[vertexSize];
	OutEdgeWeighted* edges = new OutEdgeWeighted[totalSize];
	Edge *byteEdges = new Edge[partSize];
	ull counter = 0;
	for(uint i = 0; i < vertexSize; i++)
	 	degree[i] = 0;

	cout << "calculate degree " << endl;
	ifstream infile(input, ios::in | ios::binary);
	infile.seekg(0, std::ios::end);
	ull size = infile.tellg();
	totalSize = size / sizeof(Edge);
	cout << "total edge Size " << totalSize << endl;
	infile.clear();
	infile.seekg(0);
	uint maxNode = 0;

	while(counter < totalSize) {
		cout << counter << " to " << counter + partSize << endl;
		counter += partSize;
		infile.read((char *) byteEdges, sizeof(Edge) * partSize);
		for (uint i = 0; i < partSize; ++i)
		{
			degree[byteEdges[i].source]++;
			if (maxNode < byteEdges[i].source)
			{
				maxNode = byteEdges[i].source;
			}
		}
	}
	cout << "max node is " << maxNode << endl;
	vertexSize = maxNode + 1;

	ull tempPointer = 0;
	for(uint i=0; i<vertexSize; i++)
	{
		nodePointers[i] = tempPointer;
		tempPointer += degree[i];
	}

	infile.clear();
	infile.seekg(0);

	cout << "calculate edges " << endl;
	for (uint i = 0; i < vertexSize; ++i)
	{
		nodePointersAnkor[i] = 0;
	}
	counter = 0;
	while(counter < totalSize) {
		cout << counter << " to " << counter + partSize << endl;
		counter += partSize;
		infile.read((char *) byteEdges, sizeof(Edge) * partSize);
		for (uint i = 0; i < partSize; ++i)
		{
			ull location = nodePointers[byteEdges[i].source] + nodePointersAnkor[byteEdges[i].source];
	 		edges[location].end = byteEdges[i].end;
	 		nodePointersAnkor[byteEdges[i].source]++;
	 		edges[location].w8 = rand() % 20;
		}
	}
	//delete [] nodePointersAnkor;
	cout << "degree[0] " << degree[0] << endl;
	cout << "degree[1] " << degree[1] << endl;
	//cout << "degree[50000000] " << degree[50000000] << endl;
	
	std::ofstream outfile(input.substr(0, input.length()-3)+"bwcsr", std::ofstream::binary);
	outfile.write((char*)&vertexSize, sizeof(unsigned int));
	outfile.write((char*)&totalSize, sizeof(ull));
	outfile.write ((char*)nodePointers, sizeof(ull)*vertexSize);
	outfile.write ((char*)edges, sizeof(OutEdgeWeighted)*totalSize);
	outfile.close();
	endTime = clock();
    cout << setprecision(9)  << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s | " << "S0:Prepare [edges] OK" << endl;
}


int main(int argc, char** argv)
{
	if(argc!= 2)
	{
		cout << "\nThere was an error parsing command line arguments\n";
		exit(0);
	}

	string input = string(argv[1]);
	convertByteToBCSR(input);
	//convertTxtToByte(input);
	// clock_t startTime, endTime;
	// startTime = clock();

 //    srand(0);
 //    ifstream infile;
 //    infile.open(input);
 //    stringstream ss;
 //    uint max = 0;
 //    string line;
 //    ull edgeCounter = 0;

 //    vector<EdgeWeighted> edges;
 //    EdgeWeighted newEdge,newEdge2;
 //    ull testIndex = 0;
 //    while(getline( infile, line ))
 //    {
 //    	testIndex++;
 //    	if (testIndex % 1000000000 == 1)
 //    	{
 //    		cout << testIndex / 1000000000 << " billion lines " << endl;
 //    	}
 //        ss.str("");
 //        ss.clear();
 //        ss << line;
 //        ss >> newEdge.source;
 //        ss >> newEdge.end;
 //        newEdge.w8 = rand() % 20;
 //        edges.push_back(newEdge);
 //        edgeCounter++;
 //        if(max < newEdge.source)
 //            max = newEdge.source;
 //        if(max < newEdge.end)
 //            max = newEdge.end;

 //    }
 //    infile.close();
 //    endTime = clock();
 //    cout << setprecision(9)  << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s | " << "S0:Prepare [edges] OK" << endl;

    
 //       // .bcsr
	// if(GetFileExtension(input) == "txt")
	// {
	//     cout << "A1:Begin with .el" << endl;


	// 	uint num_nodes = max + 1;
	// 	ull num_edges = edgeCounter;


	// 	ull *nodePointer = new ull[num_nodes+1];
	// 	uint *degree = new uint[num_nodes];
	// 	for(uint i=0; i<num_nodes; i++)
	// 		degree[i] = 0;
	// 	for(uint i=0; i<num_edges; i++)
	// 		degree[edges[i].source]++;
	// 	uint counter=0;
	// 	for(uint i=0; i<num_nodes; i++)
	// 	{
	// 		nodePointer[i] = counter;
	// 		counter = counter + degree[i];
	// 	}
	// 	delete[] degree;
	// 	endTime = clock();
	// 	cout << setprecision(9)  << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s | " << "A2:Prepare [nodePointer] OK" << endl;


	// 	OutEdge *edgeList = new OutEdge[num_edges];
	// 	uint *outDegreeCounter  = new uint[num_nodes];
 //        for(uint i=0; i<num_nodes; i++)
	// 		outDegreeCounter[i] = 0;
	// 	uint location;
	// 	for(ull i=0; i<num_edges; i++)
	// 	{
	// 		ull location = nodePointer[edges[i].source] + outDegreeCounter[edges[i].source];
	// 		edgeList[location].end = edges[i].end;
	// 		outDegreeCounter[edges[i].source]++;
	// 	}
	// 	delete[] outDegreeCounter;
	// 	endTime = clock();
	// 	cout << setprecision(9)  << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s | " << "A3:Prepare [edgeList] OK" << endl;


	// 	std::ofstream outfile(input.substr(0, input.length()-3)+"bcsr", std::ofstream::binary);
	// 	outfile.write((char*)&num_nodes, sizeof(unsigned int));
	// 	outfile.write((char*)&num_edges, sizeof(unsigned int));
	// 	outfile.write ((char*)nodePointer, sizeof(ull)*num_nodes);
	// 	outfile.write ((char*)edgeList, sizeof(OutEdge)*num_edges);
	// 	outfile.close();
	// 	delete[] nodePointer;
	// 	delete[] edgeList;
	// 	endTime = clock();
	// 	cout << setprecision(9)  << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s | " << "A4:End with .bcsr" << endl;
	// }



 //    // .bcsr
	// if(GetFileExtension(input) == "txt")
	// {
	//     cout << "A1:Begin with .el" << endl;


	// 	uint num_nodes = max + 1;
	// 	uint num_edges = edgeCounter;


	// 	uint *nodePointer = new uint[num_nodes+1];
	// 	uint *degree = new uint[num_nodes];
	// 	for(uint i=0; i<num_nodes; i++)
	// 		degree[i] = 0;
	// 	for(uint i=0; i<num_edges; i++)
	// 		degree[edges[i].source]++;
	// 	uint counter=0;
	// 	for(uint i=0; i<num_nodes; i++)
	// 	{
	// 		nodePointer[i] = counter;
	// 		counter = counter + degree[i];
	// 	}
	// 	delete[] degree;
	// 	endTime = clock();
	// 	cout << setprecision(9)  << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s | " << "A2:Prepare [nodePointer] OK" << endl;


	// 	OutEdge *edgeList = new OutEdge[num_edges];
	// 	uint *outDegreeCounter  = new uint[num_nodes];
 //        for(uint i=0; i<num_nodes; i++)
	// 		outDegreeCounter[i] = 0;
	// 	uint location;
	// 	for(uint i=0; i<num_edges; i++)
	// 	{
	// 		uint location = nodePointer[edges[i].source] + outDegreeCounter[edges[i].source];
	// 		edgeList[location].end = edges[i].end;
	// 		outDegreeCounter[edges[i].source]++;
	// 	}
	// 	delete[] outDegreeCounter;
	// 	endTime = clock();
	// 	cout << setprecision(9)  << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s | " << "A3:Prepare [edgeList] OK" << endl;


	// 	std::ofstream outfile(input.substr(0, input.length()-3)+"bcsr", std::ofstream::binary);
	// 	outfile.write((char*)&num_nodes, sizeof(unsigned int));
	// 	outfile.write((char*)&num_edges, sizeof(unsigned int));
	// 	outfile.write ((char*)nodePointer, sizeof(unsigned int)*num_nodes);
	// 	outfile.write ((char*)edgeList, sizeof(OutEdge)*num_edges);
	// 	outfile.close();
	// 	delete[] nodePointer;
	// 	delete[] edgeList;
	// 	endTime = clock();
	// 	cout << setprecision(9)  << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s | " << "A4:End with .bcsr" << endl;
	// }

	// // .bwcsr
	// if(GetFileExtension(input) == "txt")
	// {
	//     cout << "B1:Begin with .wel" << endl;


	// 	uint num_nodes = max + 1;
	// 	uint num_edges = edgeCounter;


	// 	uint *nodePointer = new uint[num_nodes+1];
	// 	uint *degree = new uint[num_nodes];
	// 	for(uint i=0; i<num_nodes; i++)
	// 		degree[i] = 0;
	// 	for(uint i=0; i<num_edges; i++)
	// 		degree[edges[i].source]++;
	// 	uint counter=0;
	// 	for(uint i=0; i<num_nodes; i++)
	// 	{
	// 		nodePointer[i] = counter;
	// 		counter = counter + degree[i];
	// 	}
	// 	delete[] degree;
	// 	endTime = clock();
	// 	cout << setprecision(9)  << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s | " << "B2:Prepare [nodePointer] OK" << endl;


	// 	OutEdgeWeighted *edgeList = new OutEdgeWeighted[num_edges];
	// 	uint *outDegreeCounter  = new uint[num_nodes];
 //        for(uint i=0; i<num_nodes; i++)
	// 		outDegreeCounter[i] = 0;
	// 	uint location;
	// 	for(uint i=0; i<num_edges; i++)
	// 	{
	// 		uint location = nodePointer[edges[i].source] + outDegreeCounter[edges[i].source];
	// 		edgeList[location].end = edges[i].end;
	// 		edgeList[location].w8 = edges[i].w8;
	// 		outDegreeCounter[edges[i].source]++;
	// 	}
	// 	delete[] outDegreeCounter;
	// 	endTime = clock();
	// 	cout << setprecision(9)  << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s | " << "B3:Prepare [edgeList] OK" << endl;


	// 	std::ofstream outfile(input.substr(0, input.length()-3)+"bwcsr", std::ofstream::binary);
	// 	outfile.write((char*)&num_nodes, sizeof(unsigned int));
	// 	outfile.write((char*)&num_edges, sizeof(unsigned int));
	// 	outfile.write ((char*)nodePointer, sizeof(unsigned int)*num_nodes);
	// 	outfile.write ((char*)edgeList, sizeof(OutEdgeWeighted)*num_edges);
	// 	outfile.close();
	// 	delete[] nodePointer;
	// 	delete[] edgeList;
	// 	endTime = clock();
	// 	cout << setprecision(9)  << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s | " << "B4:End with .bwcsr" << endl;
 //    }

 //    // .bcsc
	// if(GetFileExtension(input) == "txt")
	// {
	//     cout << "C1:Begin with .cl" << endl;


	// 	uint num_nodes = max + 1;
	// 	uint num_edges = edgeCounter;


	// 	uint *outDegree = new uint[num_nodes];
	// 	for(uint i=0; i<num_nodes; i++)
	// 		outDegree[i] = 0;
	// 	for(uint i=0; i<num_edges; i++)
	// 		outDegree[edges[i].source]++;
 //        cout << setprecision(9)  << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s | " << "C2:Prepare [outDegree] OK" << endl;


	// 	uint *nodePointer = new uint[num_nodes+1];
	// 	uint *inDegree = new uint[num_nodes];
	// 	for(uint i=0; i<num_nodes; i++)
	// 		inDegree[i] = 0;
	// 	for(uint i=0; i<num_edges; i++)
	// 		inDegree[edges[i].end]++;
	// 	uint counter=0;
	// 	for(uint i=0; i<num_nodes; i++)
	// 	{
	// 		nodePointer[i] = counter;
	// 		counter = counter + inDegree[i];
	// 	}
	// 	delete[] inDegree;
	// 	endTime = clock();
	// 	cout << setprecision(9)  << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s | " << "C3:Prepare [nodePointer] OK" << endl;


	// 	OutEdge *edgeList = new OutEdge[num_edges];
	// 	uint *inDegreeCounter  = new uint[num_nodes];
 //        for(uint i=0; i<num_nodes; i++)
	// 		inDegreeCounter[i] = 0;
	// 	uint location;
	// 	for(uint i=0; i<num_edges; i++)
	// 	{

	// 		uint location = nodePointer[edges[i].end] + inDegreeCounter[edges[i].end];
	// 		edgeList[location].end = edges[i].source;
	// 		inDegreeCounter[edges[i].end]++;
	// 	}
	// 	delete[] inDegreeCounter;
	// 	endTime = clock();
	// 	cout << setprecision(9)  << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s | " << "C4:Prepare [edgeList] OK" << endl;


	// 	std::ofstream outfile(input.substr(0, input.length()-3)+"bcsc", std::ofstream::binary);
	// 	outfile.write((char*)&num_nodes, sizeof(unsigned int));
	// 	outfile.write((char*)&num_edges, sizeof(unsigned int));
	// 	outfile.write ((char*)outDegree, sizeof(unsigned int)*num_nodes);
	// 	outfile.write ((char*)nodePointer, sizeof(unsigned int)*num_nodes);
	// 	outfile.write ((char*)edgeList, sizeof(OutEdge)*num_edges);
	// 	outfile.close();
	// 	delete[] outDegree;
	// 	delete[] nodePointer;
	// 	delete[] edgeList;
	// 	endTime = clock();
	// 	cout << setprecision(9)  << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s | " << "C5:End with .bcsc" << endl;
 //    }

 //    // .llbcsr
	// if(GetFileExtension(input) == "txt")
	// {
	//     cout << "D1:Begin with .el" << endl;


	// 	ull num_nodes = max + 1;
	// 	ull num_edges = edgeCounter;


	// 	ull *nodePointer = new ull[num_nodes+1];
	// 	ull *degree = new ull[num_nodes];
	// 	for(ull i=0; i<num_nodes; i++)
	// 		degree[i] = 0;
	// 	for(ull i=0; i<num_edges; i++)
	// 		degree[edges[i].source]++;
	// 	ull counter=0;
	// 	for(ull i=0; i<num_nodes; i++)
	// 	{
	// 		nodePointer[i] = counter;
	// 		counter = counter + degree[i];
	// 	}
	// 	delete[] degree;
	// 	endTime = clock();
	// 	cout << setprecision(9)  << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s | " << "D2:Prepare [nodePointer] OK" << endl;


	// 	llOutEdge *edgeList = new llOutEdge[num_edges];
	// 	ull *outDegreeCounter  = new ull[num_nodes];
 //        for(ull i=0; i<num_nodes; i++)
	// 		outDegreeCounter[i] = 0;
	// 	ull location;
	// 	for(ull i=0; i<num_edges; i++)
	// 	{
	// 		ull location = nodePointer[edges[i].source] + outDegreeCounter[edges[i].source];
	// 		edgeList[location].end = edges[i].end;
	// 		outDegreeCounter[edges[i].source]++;
	// 	}
	// 	delete[] outDegreeCounter;
	// 	endTime = clock();
	// 	cout << setprecision(9)  << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s | " << "D3:Prepare [edgeList] OK" << endl;


	// 	std::ofstream outfile(input.substr(0, input.length()-3)+"llbcsr", std::ofstream::binary);
	// 	outfile.write((char*)&num_nodes, sizeof(unsigned long long));
	// 	outfile.write((char*)&num_edges, sizeof(unsigned long long));
	// 	outfile.write ((char*)nodePointer, sizeof(unsigned long long)*num_nodes);
	// 	outfile.write ((char*)edgeList, sizeof(llOutEdge)*num_edges);
	// 	outfile.close();
	// 	delete[] nodePointer;
	// 	delete[] edgeList;
	// 	endTime = clock();
	// 	cout << setprecision(9)  << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s | " << "D4:End with .llbcsr" << endl;
	// }

	// // .llbwcsr
	// if(GetFileExtension(input) == "txt")
	// {
	//     cout << "E1:Begin with .wel" << endl;


	// 	ull num_nodes = max + 1;
	// 	ull num_edges = edgeCounter;


	// 	ull *nodePointer = new ull[num_nodes+1];
	// 	ull *degree = new ull[num_nodes];
	// 	for(ull i=0; i<num_nodes; i++)
	// 		degree[i] = 0;
	// 	for(ull i=0; i<num_edges; i++)
	// 		degree[edges[i].source]++;
	// 	ull counter=0;
	// 	for(ull i=0; i<num_nodes; i++)
	// 	{
	// 		nodePointer[i] = counter;
	// 		counter = counter + degree[i];
	// 	}
	// 	delete[] degree;
	// 	endTime = clock();
	// 	cout << setprecision(9)  << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s | " << "E2:Prepare [nodePointer] OK" << endl;


	// 	llOutEdgeWeighted *edgeList = new llOutEdgeWeighted[num_edges];
	// 	ull *outDegreeCounter  = new ull[num_nodes];
 //        for(ull i=0; i<num_nodes; i++)
	// 		outDegreeCounter[i] = 0;
	// 	ull location;
	// 	for(ull i=0; i<num_edges; i++)
	// 	{
	// 		ull location = nodePointer[edges[i].source] + outDegreeCounter[edges[i].source];
	// 		edgeList[location].end = edges[i].end;
	// 		edgeList[location].w8 = edges[i].w8;
	// 		outDegreeCounter[edges[i].source]++;
	// 	}
	// 	delete[] outDegreeCounter;
	// 	endTime = clock();
	// 	cout << setprecision(9)  << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s | " << "E3:Prepare [edgeList] OK" << endl;


	// 	std::ofstream outfile(input.substr(0, input.length()-3)+"llbwcsr", std::ofstream::binary);
	// 	outfile.write((char*)&num_nodes, sizeof(unsigned long long));
	// 	outfile.write((char*)&num_edges, sizeof(unsigned long long));
	// 	outfile.write ((char*)nodePointer, sizeof(unsigned long long)*num_nodes);
	// 	outfile.write ((char*)edgeList, sizeof(llOutEdgeWeighted)*num_edges);
	// 	outfile.close();
	// 	delete[] nodePointer;
	// 	delete[] edgeList;
	// 	endTime = clock();
	// 	cout << setprecision(9)  << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s | " << "E4:End with .llbwcsr" << endl;
 //    }

 //    // .llbcsc
	// if(GetFileExtension(input) == "txt")
	// {
	//     cout << "F1:Begin with .cl" << endl;


	// 	ull num_nodes = max + 1;
	// 	ull num_edges = edgeCounter;


	// 	ull *outDegree = new ull[num_nodes];
	// 	for(ull i=0; i<num_nodes; i++)
	// 		outDegree[i] = 0;
	// 	for(ull i=0; i<num_edges; i++)
	// 		outDegree[edges[i].source]++;
 //        cout << setprecision(9)  << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s | " << "F2:Prepare [outDegree] OK" << endl;


	// 	ull *nodePointer = new ull[num_nodes+1];
	// 	ull *inDegree = new ull[num_nodes];
	// 	for(ull i=0; i<num_nodes; i++)
	// 		inDegree[i] = 0;
	// 	for(ull i=0; i<num_edges; i++)
	// 		inDegree[edges[i].end]++;
	// 	ull counter=0;
	// 	for(ull i=0; i<num_nodes; i++)
	// 	{
	// 		nodePointer[i] = counter;
	// 		counter = counter + inDegree[i];
	// 	}
	// 	delete[] inDegree;
	// 	endTime = clock();
	// 	cout << setprecision(9)  << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s | " << "F3:Prepare [nodePointer] OK" << endl;


	// 	llOutEdge *edgeList = new llOutEdge[num_edges];
	// 	ull *inDegreeCounter  = new ull[num_nodes];
 //        for(ull i=0; i<num_nodes; i++)
	// 		inDegreeCounter[i] = 0;
	// 	ull location;
	// 	for(ull i=0; i<num_edges; i++)
	// 	{
	// 		ull location = nodePointer[edges[i].end] + inDegreeCounter[edges[i].end];
	// 		edgeList[location].end = edges[i].source;
	// 		inDegreeCounter[edges[i].end]++;
	// 	}
	// 	delete[] inDegreeCounter;
	// 	endTime = clock();
	// 	cout << setprecision(9)  << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s | " << "F4:Prepare [edgeList] OK" << endl;


	// 	std::ofstream outfile(input.substr(0, input.length()-3)+"llbcsc", std::ofstream::binary);
	// 	outfile.write((char*)&num_nodes, sizeof(unsigned long long));
	// 	outfile.write((char*)&num_edges, sizeof(unsigned long long));
	// 	outfile.write ((char*)outDegree, sizeof(unsigned long long)*num_nodes);
	// 	outfile.write ((char*)nodePointer, sizeof(unsigned long long)*num_nodes);
	// 	outfile.write ((char*)edgeList, sizeof(llOutEdge)*num_edges);
	// 	outfile.close();
	// 	delete[] outDegree;
	// 	delete[] nodePointer;
	// 	delete[] edgeList;
	// 	endTime = clock();
	// 	cout << setprecision(9)  << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s | " << "F5:End with .llbcsc" << endl;
 //    }


}
