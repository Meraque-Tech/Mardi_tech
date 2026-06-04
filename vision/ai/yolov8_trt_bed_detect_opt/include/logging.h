#ifndef TENSORRT_LOGGING_H
#define TENSORRT_LOGGING_H

#include <NvInfer.h>
#include <iostream>

class Logger : public nvinfer1::ILogger {
public:
    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override {
        if (severity <= nvinfer1::ILogger::Severity::kWARNING)
            std::cout << "[TRT] " << msg << std::endl;
    }

    void setReportableSeverity(nvinfer1::ILogger::Severity) {}
};

#endif // TENSORRT_LOGGING_H
