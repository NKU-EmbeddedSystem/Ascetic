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
	cout<<"convertTxtToByte"<<endl;
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
        //cout << newEdge.source << "  " << newEdge.end << endl;
        edges.push_back(newEdge);
        edgeCounter++;
        if(max < newEdge.source)
            max = newEdge.source;
		
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
	ull vertexSize = 1000000000;
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
		ull readsize;
		if(counter<totalSize)
		readsize=partSize;
		else
		readsize=partSize+totalSize-counter;
		cout<<"readsize: "<<readsize<<endl;
		infile.read((char *) byteEdges, sizeof(Edge) * readsize);
		for (uint i = 0; i < readsize; ++i)
		{
			degree[byteEdges[i].source]++;
			if (maxNode < byteEdges[i].source)
			{
				maxNode = byteEdges[i].source;
			}
			if (maxNode < byteEdges[i].end)
			{
				maxNode = byteEdges[i].end;
			}
		}
	}
	cout << "max node is " << maxNode << endl;
	vertexSize = maxNode + 1;
	ull tempPointer = 0;
	for(ull i=0; i<vertexSize; i++)
	{
		nodePointers[i] = tempPointer;
		if(nodePointers[i]>=totalSize)
		{
			cout<<"ndoe error at "<<i<<" nodePointers: "<<nodePointers[i]<<endl;
			getchar();
		}
		tempPointer += degree[i];
		if(tempPointer==totalSize)
		tempPointer--;
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
		ull readsize;
		if(counter<totalSize)
		readsize=partSize;
		else
		readsize=partSize+totalSize-counter;
		cout<<"readsize: "<<readsize<<endl;
		infile.read((char *) byteEdges, sizeof(Edge) * readsize);
		for (uint i = 0; i < readsize; ++i)
		{
			ull location = nodePointers[byteEdges[i].source] + nodePointersAnkor[byteEdges[i].source];
	 		edges[location].end = byteEdges[i].end;
	 		nodePointersAnkor[byteEdges[i].source]++;
		}
	}
	//delete [] nodePointersAnkor;
	for(ull i=0;i<totalSize;i++)
	{
		if(edges[i].end>=vertexSize)
		{
			cout<<"edge error at "<<i<<" with "<<edges[i].end<<endl;
			getchar();
		}
	}
	
	std::ofstream outfile(input.substr(0, input.length()-3)+"bcsr", std::ofstream::binary);
	outfile.write((char*)&vertexSize, sizeof(ull));
	outfile.write((char*)&totalSize, sizeof(ull));
	outfile.write ((char*)nodePointers, sizeof(ull)*vertexSize);
	outfile.write ((char*)edges, sizeof(OutEdge)*totalSize);
	outfile.close();
	//cout << "ull size is " << sizeof(ull) << endl;
	endTime = clock();
    cout << setprecision(9)  << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s | " << "S0:Prepare [edges] OK" << endl;
}

void convertByteToBCSC(string input) {
	cout << "convertByteToBCSC" << endl;
	clock_t startTime, endTime;
	startTime = clock();
	ull totalSize = 10000000000;
	uint partSize = 1000000000;
	uint vertexSize = 1000000000;
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
	ull inEdgesize=0;
	while(counter < totalSize) {
		cout << counter << " to " << counter + partSize << endl;
		counter += partSize;
		ull readsize;
		if(counter<totalSize)
		readsize=partSize;
		else
		readsize=partSize+totalSize-counter;
		cout<<"readsize: "<<readsize<<endl;
		infile.read((char *) byteEdges, sizeof(Edge) * readsize);
		for (uint i = 0; i < readsize; ++i)
		{
			degree[byteEdges[i].source]++;
			inDegree[byteEdges[i].end]++;
			inEdgesize++;
			if (maxNode < byteEdges[i].source)
			{
				maxNode = byteEdges[i].source;
			}
			if (maxNode < byteEdges[i].end)
			{
				maxNode = byteEdges[i].end;
			}
		}
	}
	cout << "max node is " << maxNode << endl;
	vertexSize = maxNode + 1;
	ull tempPointer = 0;
	for(uint i=0; i<vertexSize; i++)
	{
		nodePointers[i] = tempPointer;
		if(nodePointers[i]>=inEdgesize)
		{
			cout<<"ndoe error at "<<i<<" nodePointers: "<<nodePointers[i]<<endl;
			getchar();
		}
		tempPointer += inDegree[i];
		if(tempPointer==inEdgesize)
		tempPointer--;
	}
	cout << "finish calculate nodePointers " << endl;

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
		ull readsize;
		if(counter<totalSize)
		readsize=partSize;
		else
		readsize=partSize+totalSize-counter;
		cout<<"readsize: "<<readsize<<endl;
		infile.read((char *) byteEdges, sizeof(Edge) * readsize);
		for (uint i = 0; i < readsize; ++i)
		{
			ull location = nodePointers[byteEdges[i].end] + nodePointersAnkor[byteEdges[i].end];
	 		edges[location].end = byteEdges[i].source;
	 		nodePointersAnkor[byteEdges[i].end]++;
		}
	}
	//delete [] nodePointersAnkor;
	for(ull i=0;i<inEdgesize;i++)
	{
		if(edges[i].end>=inEdgesize)
		{
			cout<<"edge error at "<<i<<" with "<<edges[i].end<<endl;
			getchar();
		}
	}

	std::ofstream outfile(input.substr(0, input.length()-3)+"bcsc", std::ofstream::binary);
	outfile.write((char*)&vertexSize, sizeof(ull));
	outfile.write((char*)&totalSize, sizeof(ull));
	outfile.write ((char*)degree, sizeof(uint)*vertexSize);
	outfile.write ((char*)nodePointers, sizeof(ull)*vertexSize);
	outfile.write ((char*)edges, sizeof(OutEdge)*inEdgesize);
	outfile.close();
	endTime = clock();
    cout << setprecision(9)  << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s | " << "S0:Prepare [edges] OK" << endl;
}

void convertByteToBWCSR(string input) {
	cout << "convertByteToBWCSR" << endl;
	srand(0);
	clock_t startTime, endTime;
	startTime = clock();
	ull totalSize = 10000000000;
	uint partSize = 1000000000;
	ull vertexSize = 1000000000;
	ull* nodePointers = new ull[vertexSize];
	ull* nodePointersAnkor = new ull[vertexSize];
	ull* degree = new ull[vertexSize];
	OutEdgeWeighted* edges = new OutEdgeWeighted[totalSize];
	Edge *byteEdges = new Edge[partSize];
	ull counter = 0;
	for(ull i = 0; i < vertexSize; i++)
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
		ull readsize;
		if(counter<totalSize)
		readsize=partSize;
		else
		readsize=partSize+totalSize-counter;
		cout<<"readsize: "<<readsize<<endl;
		infile.read((char *) byteEdges, sizeof(Edge) * readsize);
		for (uint i = 0; i < readsize; ++i)
		{
			degree[byteEdges[i].source]++;
			if (maxNode < byteEdges[i].source)
			{
				maxNode = byteEdges[i].source;
			}
			if (maxNode < byteEdges[i].end)
			{
				maxNode = byteEdges[i].end;
			}
		}
	}
	cout << "max node is " << maxNode << endl;
	vertexSize = maxNode + 1;
	ull tempPointer = 0;
	for(ull i=0; i<vertexSize; i++)
	{
		nodePointers[i] = tempPointer;
		if(nodePointers[i]>=totalSize)
		{
			cout<<"ndoe error at "<<i<<" nodePointers: "<<nodePointers[i]<<endl;
			getchar();
		}
		tempPointer += degree[i];
		if(tempPointer==totalSize)
		tempPointer--;
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
		ull readsize;
		if(counter<totalSize)
		readsize=partSize;
		else
		readsize=partSize+totalSize-counter;
		cout<<"readsize: "<<readsize<<endl;
		infile.read((char *) byteEdges, sizeof(Edge) * partSize);
		for (uint i = 0; i < partSize; ++i)
		{
			ull location = nodePointers[byteEdges[i].source] + nodePointersAnkor[byteEdges[i].source];
	 		edges[location].end = byteEdges[i].end;
	 		nodePointersAnkor[byteEdges[i].source]++;
	 		edges[location].w8 = rand() % 20;
		}
	}
	for(ull i=0;i<totalSize;i++)
	{
		if(edges[i].end>=vertexSize)
		{
			cout<<"edge error at "<<i<<" with "<<edges[i].end<<endl;
			getchar();
		}
	}
	//delete [] nodePointersAnkor;
	cout << "degree[0] " << degree[0] << endl;
	cout << "degree[1] " << degree[1] << endl;
	//cout << "degree[50000000] " << degree[50000000] << endl;
	
	std::ofstream outfile(input.substr(0, input.length()-3)+"bwcsr", std::ofstream::binary);
	outfile.write((char*)&vertexSize, sizeof(ull));
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
	//convertTxtToByte(input);
	//convertByteToBCSR(input);
	//convertByteToBCSC(input);
	convertByteToBWCSR(input); 
	

}
