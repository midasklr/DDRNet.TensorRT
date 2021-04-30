#include <iostream>
#include <chrono>
#include "cuda_runtime_api.h"
#include "logging.h"
#include "common.hpp"
#include <math.h>
#include "calibrator.h"


//#define USE_INT8  // comment out this if want to use INT8
#define USE_FP16  // comment out this if want to use FP32
#define DEVICE 0  // GPU id
static const int INPUT_H = 1024;
static const int INPUT_W = 1024;
static const int OUT_MAP_H = 128;
static const int OUT_MAP_W = 128;
const char* INPUT_BLOB_NAME = "input_0";
const char* OUTPUT_BLOB_NAME = "output_0";
static Logger gLogger;

// Creat the engine using only the API and not any parser.
ICudaEngine* createEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt) {
    INetworkDefinition* network = builder->createNetworkV2(0U);
    // Create input tensor of shape {1, 3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims4{ 1, 3, INPUT_H, INPUT_W });
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights("../DDRNet_CS.wts");
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };

    IConvolutionLayer* conv1 = network->addConvolution(*data, 32, DimsHW{ 3, 3 }, weightMap["conv1.0.weight"], weightMap["conv1.0.bias"]);
    assert(conv1);
    conv1->setStride(DimsHW{ 2, 2 });
    conv1->setPadding(DimsHW{ 1, 1 });

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), "conv1.1", 1e-5);

    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);

    IConvolutionLayer* conv2 = network->addConvolution(*relu1->getOutput(0), 32, DimsHW{ 3, 3 }, weightMap["conv1.3.weight"], weightMap["conv1.3.bias"]);
    assert(conv2);
    conv2->setStride(DimsHW{ 2, 2 });
    conv2->setPadding(DimsHW{ 1, 1 });

    IScaleLayer* bn2 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), "conv1.4", 1e-5);

    IActivationLayer* relu2 = network->addActivation(*bn2->getOutput(0), ActivationType::kRELU);
    assert(relu2);

    // layer1
    ILayer* layer1_0 = basicBlock(network, weightMap, *relu2->getOutput(0), 32, 32, 1, false, false, "layer1.0.");
    ILayer* layer1_1 = basicBlock(network, weightMap, *layer1_0->getOutput(0), 32, 32, 1, false, true, "layer1.1.");
    IActivationLayer* layer1_relu = network->addActivation(*layer1_1->getOutput(0), ActivationType::kRELU);
    assert(layer1_relu);

    // layer2
    ILayer* layer2_0 = basicBlock(network, weightMap, *layer1_relu->getOutput(0), 32, 64, 2, true, false, "layer2.0.");
    ILayer* layer2_1 = basicBlock(network, weightMap, *layer2_0->getOutput(0), 64, 64, 1, false, true, "layer2.1."); // 1/8
    IActivationLayer* layer2_relu = network->addActivation(*layer2_1->getOutput(0), ActivationType::kRELU);
    assert(layer2_relu);

    // layer3
    ILayer* layer3_0 = basicBlock(network, weightMap, *layer2_relu->getOutput(0), 64, 128, 2, true, false, "layer3.0.");
    ILayer* layer3_1 = basicBlock(network, weightMap, *layer3_0->getOutput(0), 128, 128, 1, false, true, "layer3.1."); // 1/16
    IActivationLayer* layer3_relu = network->addActivation(*layer3_1->getOutput(0), ActivationType::kRELU);
    assert(layer3_relu);   // layer[2]

    // x_ = self.layer3_(self.relu(layers[1]))
    // layer3_
    ILayer* layer3_10 = basicBlock(network, weightMap, *layer2_relu->getOutput(0), 64, 64, 1, false, false, "layer3_.0.");
    ILayer* layer3_11 = basicBlock(network, weightMap, *layer3_10->getOutput(0), 64, 64, 1, false, true, "layer3_.1."); // x_ = self.layer3_(self.relu(layers[1]))
    //

    // down3
    IActivationLayer* down3_input_relu = network->addActivation(*layer3_11->getOutput(0), ActivationType::kRELU);
    assert(down3_input_relu);

    ILayer* down3_out = down3(network, weightMap, *down3_input_relu->getOutput(0), 128, "down3.");
    //  x = x + self.down3(self.relu(x_))

    IElementWiseLayer* down3_add = network->addElementWise(*layer3_1->getOutput(0), *down3_out->getOutput(0), ElementWiseOperation::kSUM);

    //x_ = x_ + F.interpolate(self.compression3(self.relu(layers[2])), size=[height_output, width_output], mode='bilinear',align_corners=True)
    ILayer* compression3_input = compression3(network, weightMap, *layer3_relu->getOutput(0), 64, "compression3.");

    float *deval = reinterpret_cast<float*>(malloc(sizeof(float) * 64 * 2 * 2));
    for (int i = 0; i < 64 * 2 * 2; i++) {
        deval[i] = 1.0;
    }
    Weights deconvwts1{ DataType::kFLOAT, deval, 64 * 2 * 2 };
    IDeconvolutionLayer* compression3_up = network->addDeconvolutionNd(*compression3_input->getOutput(0), 64, DimsHW{ 2, 2 }, deconvwts1, emptywts);
    compression3_up->setStrideNd(DimsHW{ 2, 2 });
    compression3_up->setNbGroups(64);
    IElementWiseLayer* compression3_add = network->addElementWise(*layer3_11->getOutput(0), *compression3_up->getOutput(0), ElementWiseOperation::kSUM);
//  x_ = self.layer4_(self.relu(x_))
    // layer4
    IActivationLayer* layer4_input = network->addActivation(*down3_add->getOutput(0), ActivationType::kRELU);
    ILayer* layer4_0 = basicBlock(network, weightMap, *layer4_input->getOutput(0), 128, 256, 2, true, false, "layer4.0.");
    //  x = self.layer4(self.relu(x))
    ILayer* layer4_1 = basicBlock(network, weightMap, *layer4_0->getOutput(0), 256, 256, 1, false, true, "layer4.1."); // 1/32
    IActivationLayer* layer4_relu = network->addActivation(*layer4_1->getOutput(0), ActivationType::kRELU);
    assert(layer4_relu);

    // layer4_
    IActivationLayer* layer4_1_input = network->addActivation(*compression3_add->getOutput(0), ActivationType::kRELU);
    ILayer* layer4_10 = basicBlock(network, weightMap, *layer4_1_input->getOutput(0), 64, 64, 1, false, false, "layer4_.0.");
    //  x_ = self.layer4_(self.relu(x_))
    ILayer* layer4_11 = basicBlock(network, weightMap, *layer4_10->getOutput(0), 64, 64, 1, false, true, "layer4_.1."); // 1/8
    // down4
    IActivationLayer* down4_input_relu = network->addActivation(*layer4_11->getOutput(0), ActivationType::kRELU);
    assert(down4_input_relu);
    ILayer* down4_out = down4(network, weightMap, *down4_input_relu->getOutput(0), 128, "down4.");

    IElementWiseLayer* down4_add = network->addElementWise(*layer4_1->getOutput(0), *down4_out->getOutput(0), ElementWiseOperation::kSUM);
//         x_ = x_ + F.interpolate(self.compression4(self.relu(layers[3])),size=[height_output, width_output],mode='bilinear',align_corners=True)
    ILayer* compression4_input = compression4(network, weightMap, *layer4_relu->getOutput(0), 64, "compression4.");

    float *deval2 = reinterpret_cast<float*>(malloc(sizeof(float) * 64 * 4 * 4));
    for (int i = 0; i < 64 * 4 * 4; i++) {
        deval2[i] = 1.0;
    }
    Weights deconvwts2{ DataType::kFLOAT, deval2, 64 * 4 * 4 };
    IDeconvolutionLayer* compression4_up = network->addDeconvolutionNd(*compression4_input->getOutput(0), 64, DimsHW{ 4, 4 }, deconvwts2, emptywts);
    compression4_up->setStrideNd(DimsHW{ 4, 4 });
    compression4_up->setNbGroups(64);

    IElementWiseLayer* compression4_add = network->addElementWise(*layer4_11->getOutput(0), *compression4_up->getOutput(0), ElementWiseOperation::kSUM);
    IActivationLayer* compression4_add_relu = network->addActivation(*compression4_add->getOutput(0), ActivationType::kRELU);
    assert(compression4_add_relu);
    // layer5_
    //  x_ = self.layer5_(self.relu(x_))
    ILayer* layer5_ = Bottleneck(network, weightMap, *compression4_add_relu->getOutput(0), 64, 64, 1, true, true, "layer5_.0.");

    // layer5
    IActivationLayer* layer5_input = network->addActivation(*down4_add->getOutput(0), ActivationType::kRELU);
    assert(layer5_input);
    ILayer* layer5 = Bottleneck(network, weightMap, *layer5_input->getOutput(0), 256, 256, 2, true, true, "layer5.0.");
    ILayer* ssp = DAPPM(network, weightMap, *layer5->getOutput(0), 512, 128, 128, "spp.");

    float *deval3 = reinterpret_cast<float*>(malloc(sizeof(float) * 128 * 8 * 8));
    for (int i = 0; i < 128 * 8 * 8; i++) {
        deval3[i] = 1.0;
    }
    Weights deconvwts3{ DataType::kFLOAT, deval3, 128 * 8 * 8 };
    IDeconvolutionLayer* spp_up = network->addDeconvolutionNd(*ssp->getOutput(0), 128, DimsHW{ 8, 8 }, deconvwts3, emptywts);
    spp_up->setStrideNd(DimsHW{ 8, 8 });
    spp_up->setNbGroups(128);
    // x_ = self.final_layer(x + x_)

    IElementWiseLayer* final_in = network->addElementWise(*spp_up->getOutput(0), *layer5_->getOutput(0), ElementWiseOperation::kSUM);

    ILayer* seg_out= segmenthead(network, weightMap, *final_in->getOutput(0), 64, 19, "final_layer.");

//    IActivationLayer* thresh = network->addActivation(*seg_out->getOutput(0), ActivationType::kSIGMOID);
//    assert(thresh);

    // y = F.interpolate(y, size=(H, W)) 
    seg_out->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*seg_out->getOutput(0));

//    IOptimizationProfile* profile = builder->createOptimizationProfile();
//    profile->setDimensions(INPUT_BLOB_NAME, OptProfileSelector::kMIN, Dims4(1, 3, MIN_INPUT_SIZE, MIN_INPUT_SIZE));
//    profile->setDimensions(INPUT_BLOB_NAME, OptProfileSelector::kOPT, Dims4(1, 3, OPT_INPUT_H, OPT_INPUT_W));
//    profile->setDimensions(INPUT_BLOB_NAME, OptProfileSelector::kMAX, Dims4(1, 3, MAX_INPUT_SIZE, MAX_INPUT_SIZE));
//    config->addOptimizationProfile(profile);

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB

#if defined(USE_INT8)
    std::cout << "Your platform support int8: " << (builder->platformHasFastInt8() ? "true" : "false") << std::endl;
    assert(builder->platformHasFastInt8());
    config->setFlag(BuilderFlag::kINT8);
    Int8EntropyCalibrator2 *calibrator = new Int8EntropyCalibrator2(1, INPUT_W, INPUT_H, "../calib/", "int8calib.table", INPUT_BLOB_NAME);
    config->setInt8Calibrator(calibrator);
#elif defined(USE_FP16)
    config->setFlag(BuilderFlag::kFP16);
#endif
    std::cout << "Building engine, please wait for a while..." << std::endl;
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap) {
        free((void*)(mem.second.values));
    }

    return engine;
}

void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream) {
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine* engine = createEngine(maxBatchSize, builder, config, DataType::kFLOAT);
    //ICudaEngine* engine = createEngine(maxBatchSize, builder, config, DataType::kFLOAT);
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
}

void doInference(IExecutionContext& context, float* input, float* output) {
    const ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
//    std::cout<<"engine.getNbBindings():"<<engine.getNbBindings()<<std::endl;
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);
//    context.setBindingDimensions(inputIndex, Dims4(1, 3, input_h, input_w));

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], 3 * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex],   19* OUT_MAP_H* OUT_MAP_W * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
//    context.enqueueV2(buffers, stream, nullptr);
    context.enqueue(1, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex],   19* OUT_MAP_H* OUT_MAP_W * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

int main(int argc, char** argv) {
    cudaSetDevice(DEVICE);
    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{ nullptr };
    size_t size{ 0 };

    if (argc == 2 && std::string(argv[1]) == "-s") {
        IHostMemory* modelStream{ nullptr };
        APIToModel(1, &modelStream);
        assert(modelStream != nullptr);
        std::ofstream p("DDRNet.engine", std::ios::binary);
        if (!p) {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        modelStream->destroy();
        return 0;
    }
    else if (argc == 3 && std::string(argv[1]) == "-d") {
        std::ifstream file("DDRNet.engine", std::ios::binary);
        if (file.good()) {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
        }
    }
    else {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./debnet -s  // serialize model to plan file" << std::endl;
        std::cerr << "./debnet -d ../samples  // deserialize plan file and run inference" << std::endl;
        return -1;
    }

    // prepare input data ---------------------------
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;

    std::vector<std::string> file_names;
    if (read_files_in_dir(argv[2], file_names) < 0) {
        std::cout << "read_files_in_dir failed." << std::endl;
        return -1;
    }

    std::vector<float> mean_value{ 0.406, 0.456, 0.485 };  // BGR
    std::vector<float> std_value{ 0.225, 0.224, 0.229 };
    int fcount = 0;
    for (auto f : file_names) {
        fcount++;
        std::cout << fcount << "  " << f << std::endl;
        cv::Mat pr_img = cv::imread(std::string(argv[2]) + "/" + f);
        cv::resize(pr_img,pr_img,cv::Size(INPUT_W,INPUT_H));
        if (pr_img.empty()) continue;
        float* data = new float[3 * pr_img.rows * pr_img.cols];
        int i = 0;
        for (int row = 0; row < pr_img.rows; ++row) {
            uchar* uc_pixel = pr_img.data + row * pr_img.step;
            for (int col = 0; col < pr_img.cols; ++col) {
                data[i] = (uc_pixel[2] / 255.0 - mean_value[2]) / std_value[2];
                data[i + pr_img.rows * pr_img.cols] = (uc_pixel[1] / 255.0 - mean_value[1]) / std_value[1];
                data[i + 2 * pr_img.rows * pr_img.cols] = (uc_pixel[0] / 255.0 - mean_value[0]) / std_value[0];
                uc_pixel += 3;
                ++i;
            }
        }
        float* prob = new float[ 19* OUT_MAP_H* OUT_MAP_W];
        // Run inference
        auto start = std::chrono::system_clock::now();
        doInference(*context, data, prob);
        auto end = std::chrono::system_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

        // show mask
        cv::Mat out;
        out.create(OUT_MAP_H, OUT_MAP_W, CV_32FC(19));
        out = read2mat(prob, out);
//        cv::resize(out, real_out, real_out.size());
        cv::Mat mask;
        mask.create(OUT_MAP_H, OUT_MAP_W, CV_8UC3);
        mask = map2cityscape(out, mask);
        cv::resize(mask,mask,cv::Size(INPUT_W,INPUT_H));
        cv::Mat result;
        cv::addWeighted(pr_img,0.7,mask,0.3,1,result);
        cv::resize(result,result,cv::Size(1024,512));
        cv::imwrite("result_" + f, result);
        delete prob;
        delete data;
    }
    return 0;
}
