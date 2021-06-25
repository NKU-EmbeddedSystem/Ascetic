//
// Created by gxl on 2021/3/29.
//

#ifndef PTGRAPH_TIMERECORD_CUH
#define PTGRAPH_TIMERECORD_CUH

#include <chrono>
#include <iostream>
#include <typeinfo>
#include <string>
#include <utility>

using namespace std;

template<class TimeUnit>
class TimeRecord {
private:
    chrono::time_point<chrono::steady_clock, chrono::nanoseconds> startTime;
    long duration{};
    bool isStart = false;
    string recordName = "TimeRecord";
public:
    TimeRecord() {}

    explicit TimeRecord(string name) : recordName(std::move(name)) {

    };

    void startRecord() {
        isStart = true;
        startTime = chrono::steady_clock::now();
    }

    void endRecord() {
        if (isStart) {
            isStart = false;
            duration += chrono::duration_cast<TimeUnit>(chrono::steady_clock::now() - startTime).count();
        } else {
            duration = 0;
            printf(" No start record! this reocrd is 0! \n");
        }

    }

    void print() {
        if (typeid(TimeUnit) == typeid(chrono::milliseconds)) {
            cout << recordName << " time is " << duration << " ms " << endl;
        } else if (typeid(TimeUnit) == typeid(chrono::microseconds)) {
            cout << recordName << " time is " << duration << " micro s " << endl;
        } else if (typeid(TimeUnit) == typeid(chrono::nanoseconds)) {
            cout << recordName << " time is " << duration << " ns " << endl;
        }
    }

    void clearRecord() {
        isStart = false;
        duration = 0;
    }

};


#endif //PTGRAPH_TIMERECORD_CUH
