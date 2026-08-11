#define USE_FP16
//#define USE_INT8

const static char *kInputTensorName = "images";
const static char *kOutputTensorName = "output";
// The shared YoloLayerPlugin (plugin/yololayer.*) also serves pose models in
// its reference implementation; bed_detect never sets is_pose=true, but the
// symbol must exist for addYoLoLayer()/block.cpp to compile against it.
const static int kPoseNumClass = 1;
const static int kNumberOfPoints = 17;      // number of keypoints total (unused here, is_pose is always false)
const static float kConfThreshKeypoints = 0.5f;  // unused here, is_pose is always false
const static int kBatchSize = 1;
const static int kGpuId = 0;
// Runtime-configurable (not a compile-time constant): the "-s wts engine sub_type [H W]"
// CLI mode overrides these before building the engine. See main_fun.cpp.
extern int kInputH;
extern int kInputW;
extern int kNumClass;
const static float kNmsThresh = 0.45f;
const static float kConfThresh = 0.5f;
const static int kMaxInputImageSize = 3000 * 3000;
const static int kMaxNumOutputBbox = 1000;
// INT8 calibration image folder (only used if USE_INT8 is enabled above)
const static char* kInputQuantizationFolder = "./coco_calib";
