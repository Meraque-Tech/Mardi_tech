#pragma once

#include <assert.h>
#include <string>
#include "NvInfer.h"

// Generic gd (depth)/gw (width)/max_channels-scaled builder, matching
// Ultralytics' official YOLOv8 scaling table -- replaces the old fixed
// buildEngineYolov8n/s/m/l/x functions. See main_fun.cpp's serialize_engine()
// for the n/s/m/l/x -> (gd, gw, max_channels) lookup.
nvinfer1::IHostMemory* buildEngineYolov8Det(nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config,
                                            nvinfer1::DataType dt, const std::string& wts_path, float& gd, float& gw,
                                            int& max_channels);
