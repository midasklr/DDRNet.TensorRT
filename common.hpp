#ifndef DDRNET_COMMON_H_
#define DDRNET_COMMON_H_

#include <iostream>
#include <fstream>
#include <map>
#include <sstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "dirent.h"
#include "NvInfer.h"
#include <chrono>

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

using namespace nvinfer1;

// TensorRT weight files have a simple space delimited format:
// [type] [size] <data x size in hex>
std::map<std::string, Weights> loadWeights(const std::string file) {
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, Weights> weightMap;

    // Open weights file
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file.");

    // Read number of weight blobs
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    while (count--) {
        Weights wt{ DataType::kFLOAT, nullptr, 0 };
        uint32_t size;

        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> size;
        wt.type = DataType::kFLOAT;

        // Load blob
        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x) {
            input >> std::hex >> val[x];
        }
        wt.values = val;

        wt.count = size;
        weightMap[name] = wt;
    }

    return weightMap;
}

IScaleLayer* addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps) {
    float *gamma = (float*)weightMap[lname + ".weight"].values;
    float *beta = (float*)weightMap[lname + ".bias"].values;
    float *mean = (float*)weightMap[lname + ".running_mean"].values;
    float *var = (float*)weightMap[lname + ".running_var"].values;
    int len = weightMap[lname + ".running_var"].count;

    float *scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{ DataType::kFLOAT, scval, len };

    float *shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    Weights shift{ DataType::kFLOAT, shval, len };

    float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        pval[i] = 1.0;
    }
    Weights power{ DataType::kFLOAT, pval, len };

    weightMap[lname + ".scale"] = scale;
    weightMap[lname + ".shift"] = shift;
    weightMap[lname + ".power"] = power;
    IScaleLayer* scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale_1);
    return scale_1;
}


ILayer* basicBlock(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int inch, int outch, int stride, bool downsample, bool no_relu, std::string lname) {
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };

    IConvolutionLayer* conv1 = network->addConvolution(input, outch, DimsHW{ 3, 3 }, weightMap[lname + "conv1.weight"], emptywts);
    assert(conv1);
    conv1->setStride(DimsHW{ stride, stride });
    conv1->setPadding(DimsHW{ 1, 1 });

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "bn1", 1e-5);

    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);

    IConvolutionLayer* conv2 = network->addConvolution(*relu1->getOutput(0), outch, DimsHW{ 3, 3 }, weightMap[lname + "conv2.weight"], emptywts);
    assert(conv2);
    conv2->setPadding(DimsHW{ 1, 1 });

    IScaleLayer* bn2 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), lname + "bn2", 1e-5);

    if(downsample){
        IConvolutionLayer* convdown = network->addConvolution(input, outch, DimsHW{ 1, 1 }, weightMap[lname + "downsample.0.weight"], emptywts);
        assert(convdown);
        convdown->setStride(DimsHW{ stride, stride});
        convdown->setPadding(DimsHW{ 0, 0 });

        IScaleLayer* bndown = addBatchNorm2d(network, weightMap, *convdown->getOutput(0), lname + "downsample.1", 1e-5);

        IElementWiseLayer* ew1 = network->addElementWise(*bn2->getOutput(0), *bndown->getOutput(0), ElementWiseOperation::kSUM);
        if(no_relu){
            return ew1;
        }else{
            IActivationLayer* relu3 = network->addActivation(*ew1->getOutput(0), ActivationType::kRELU);
            assert(relu3);
            return relu3;
        }
    }
    IElementWiseLayer* ew2 = network->addElementWise(input, *bn2->getOutput(0), ElementWiseOperation::kSUM);
    if(no_relu){
        return ew2;
    }else{
        IActivationLayer* relu3 = network->addActivation(*ew2->getOutput(0), ActivationType::kRELU);
        assert(relu1);
        return relu3;
    }
}

ILayer* Bottleneck(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int inch, int outch, int stride, bool downsample, bool no_relu, std::string lname) {
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };

    IConvolutionLayer* conv1 = network->addConvolution(input, outch, DimsHW{ 1, 1 }, weightMap[lname + "conv1.weight"], emptywts);
    assert(conv1);
    conv1->setStride(DimsHW{ 1, 1 });
    conv1->setPadding(DimsHW{ 0, 0 });

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "bn1", 1e-5);

    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);

    IConvolutionLayer* conv2 = network->addConvolution(*relu1->getOutput(0), outch, DimsHW{ 3, 3 }, weightMap[lname + "conv2.weight"], emptywts);
    assert(conv2);
    conv2->setStride(DimsHW{ stride, stride });
    conv2->setPadding(DimsHW{ 1, 1 });


    IScaleLayer* bn2 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), lname + "bn2", 1e-5);

    IActivationLayer* relu2 = network->addActivation(*bn2->getOutput(0), ActivationType::kRELU);
    assert(relu2);

    IConvolutionLayer* conv3 = network->addConvolution(*relu2->getOutput(0), outch*2, DimsHW{ 1, 1 }, weightMap[lname + "conv3.weight"], emptywts);
    assert(conv3);
    conv3->setStride(DimsHW{ 1, 1 });
    conv3->setPadding(DimsHW{ 0, 0 });

    IScaleLayer* bn3 = addBatchNorm2d(network, weightMap, *conv3->getOutput(0), lname + "bn3", 1e-5);

    if(downsample){
        IConvolutionLayer* convdown = network->addConvolution(input, outch*2, DimsHW{ 1, 1 }, weightMap[lname + "downsample.0.weight"], emptywts);
        assert(convdown);
        convdown->setStride(DimsHW{ stride, stride });
        conv1->setPadding(DimsHW{ 0, 0 });


        IScaleLayer* bndown = addBatchNorm2d(network, weightMap, *convdown->getOutput(0), lname + "downsample.1", 1e-5);

        IElementWiseLayer* ew1 = network->addElementWise(*bn3->getOutput(0), *bndown->getOutput(0), ElementWiseOperation::kSUM);
        if(no_relu){
            return ew1;
        }else{
            IActivationLayer* relu3 = network->addActivation(*ew1->getOutput(0), ActivationType::kRELU);
            assert(relu1);
            return relu3;
        }
    }
    IElementWiseLayer* ew2 = network->addElementWise(input, *bn3->getOutput(0), ElementWiseOperation::kSUM);
    if(no_relu){
        return ew2;
    }else{
        IActivationLayer* relu3 = network->addActivation(*ew2->getOutput(0), ActivationType::kRELU);
        assert(relu1);
        return relu3;
    }
}


ILayer* compression3(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int highres_planes, std::string lname) {
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };

    IConvolutionLayer* conv1 = network->addConvolution(input, highres_planes , DimsHW{ 1, 1 }, weightMap[lname + "0.weight"], emptywts);
    assert(conv1);
    conv1->setPadding(DimsHW{ 0, 0 });

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "1", 1e-5);

    return bn1;
}

ILayer* compression4(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int highres_planes, std::string lname) {
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };

    IConvolutionLayer* conv1 = network->addConvolution(input, highres_planes , DimsHW{ 1, 1 }, weightMap[lname + "0.weight"], emptywts);
    assert(conv1);
    conv1->setPadding(DimsHW{ 0, 0 });

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "1", 1e-5);

    return bn1;
}

ILayer* down3(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int planes, std::string lname) {
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };

    IConvolutionLayer* conv1 = network->addConvolution(input, planes , DimsHW{ 3, 3 }, weightMap[lname + "0.weight"], emptywts);
    assert(conv1);
    conv1->setStride(DimsHW{ 2, 2 });
    conv1->setPadding(DimsHW{ 1, 1 });

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "1", 1e-5);

    return bn1;
}

ILayer* down4(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int planes, std::string lname) {
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };

    IConvolutionLayer* conv1 = network->addConvolution(input, planes , DimsHW{ 3, 3 }, weightMap[lname + "0.weight"], emptywts);
    assert(conv1);
    conv1->setStride(DimsHW{ 2, 2 });
    conv1->setPadding(DimsHW{ 1, 1 });

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "1", 1e-5);

    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);

    IConvolutionLayer* conv2 = network->addConvolution(*relu1->getOutput(0), planes*2 , DimsHW{ 3, 3 }, weightMap[lname + "3.weight"], emptywts);
    assert(conv2);
    conv2->setStride(DimsHW{ 2, 2 });
    conv2->setPadding(DimsHW{ 1, 1 });

    IScaleLayer* bn2 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), lname + "4", 1e-5);

    return bn2;
}

ILayer* DAPPM(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int inplanes, int branch_planes, int outplanes, std::string lname) {
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };

    IScaleLayer* scale0bn = addBatchNorm2d(network, weightMap, input, lname + "scale0.0", 1e-5);

    IActivationLayer* scale0relu = network->addActivation(*scale0bn->getOutput(0), ActivationType::kRELU);

    IConvolutionLayer* scale0conv = network->addConvolution(*scale0relu->getOutput(0), branch_planes , DimsHW{ 1, 1 }, weightMap[lname + "scale0.2.weight"], emptywts);
    assert(scale0conv);
    scale0conv->setPadding(DimsHW{ 0, 0 });

    // x_list[1]
    IPoolingLayer* scale1pool = network->addPooling(input, PoolingType::kAVERAGE, DimsHW{ 5, 5 });
    assert(scale1pool);
    scale1pool->setStride(DimsHW{ 2, 2 });
    scale1pool->setPadding(DimsHW{ 2, 2 });

    IScaleLayer* scale1bn = addBatchNorm2d(network, weightMap, *scale1pool->getOutput(0), lname + "scale1.1", 1e-5);

    IActivationLayer* scale1relu = network->addActivation(*scale1bn->getOutput(0), ActivationType::kRELU);
    IConvolutionLayer* scale1conv = network->addConvolution(*scale1relu->getOutput(0), branch_planes , DimsHW{ 1, 1 }, weightMap[lname + "scale1.3.weight"], emptywts);
    assert(scale1conv);
    scale1conv->setPadding(DimsHW{ 0, 0 });

    float *deval = reinterpret_cast<float*>(malloc(sizeof(float) * branch_planes * 2 * 2));
    for (int i = 0; i < branch_planes * 2 * 2; i++) {
        deval[i] = 1.0;
    }
    Weights deconvwts1{ DataType::kFLOAT, deval, branch_planes * 2 * 2 };
    IDeconvolutionLayer* scale1_interpolate = network->addDeconvolutionNd(*scale1conv->getOutput(0), branch_planes, DimsHW{ 2, 2 }, deconvwts1, emptywts);
    scale1_interpolate->setStrideNd(DimsHW{ 2, 2 });
    scale1_interpolate->setNbGroups(branch_planes);

    IElementWiseLayer* process1_input = network->addElementWise(*scale1_interpolate->getOutput(0), *scale0conv->getOutput(0), ElementWiseOperation::kSUM);

    IScaleLayer* process1bn = addBatchNorm2d(network, weightMap, *process1_input->getOutput(0), lname + "process1.0", 1e-5);

    IActivationLayer* process1relu = network->addActivation(*process1bn->getOutput(0), ActivationType::kRELU);

    IConvolutionLayer* process1conv = network->addConvolution(*process1relu->getOutput(0), branch_planes , DimsHW{ 3, 3 }, weightMap[lname + "process1.2.weight"], emptywts);
    assert(process1conv);
    process1conv->setPadding(DimsHW{ 1, 1 });

    // x_list[2]
    IPoolingLayer* scale2pool = network->addPooling(input, PoolingType::kAVERAGE, DimsHW{ 9, 9 });
    assert(scale2pool);
    scale2pool->setStride(DimsHW{ 4, 4 });
    scale2pool->setPadding(DimsHW{ 4, 4 });


    IScaleLayer* scale2bn = addBatchNorm2d(network, weightMap, *scale2pool->getOutput(0), lname + "scale2.1", 1e-5);

    IActivationLayer* scale2relu = network->addActivation(*scale2bn->getOutput(0), ActivationType::kRELU);
    IConvolutionLayer* scale2conv = network->addConvolution(*scale2relu->getOutput(0), branch_planes , DimsHW{ 1, 1 }, weightMap[lname + "scale2.3.weight"], emptywts);
    assert(scale2conv);
    scale2conv->setPadding(DimsHW{ 0, 0 });

    float *deval2 = reinterpret_cast<float*>(malloc(sizeof(float) * branch_planes * 4 * 4));
    for (int i = 0; i < branch_planes * 4 * 4; i++) {
        deval2[i] = 1.0;
    }
    Weights deconvwts2{ DataType::kFLOAT, deval2, branch_planes * 4 * 4 };
    IDeconvolutionLayer* scale2_interpolate = network->addDeconvolutionNd(*scale2conv->getOutput(0), branch_planes, DimsHW{ 4, 4 }, deconvwts2, emptywts);
    scale2_interpolate->setStrideNd(DimsHW{ 4, 4 });
    scale2_interpolate->setNbGroups(branch_planes);

    IElementWiseLayer* process2_input = network->addElementWise(*scale2_interpolate->getOutput(0), *process1conv->getOutput(0), ElementWiseOperation::kSUM);
//  process2
    IScaleLayer* process2bn = addBatchNorm2d(network, weightMap, *process2_input->getOutput(0), lname + "process2.0", 1e-5);

    IActivationLayer* process2relu = network->addActivation(*process2bn->getOutput(0), ActivationType::kRELU);

    IConvolutionLayer* process2conv = network->addConvolution(*process2relu->getOutput(0), branch_planes , DimsHW{ 3, 3 }, weightMap[lname + "process2.2.weight"], emptywts);
    assert(process2conv);
    process2conv->setPadding(DimsHW{ 1, 1 });

// scale3
    IPoolingLayer* scale3pool = network->addPooling(input, PoolingType::kAVERAGE, DimsHW{ 17, 17 });
    assert(scale3pool);
    scale3pool->setStride(DimsHW{ 8, 8 });
    scale3pool->setPadding(DimsHW{ 8, 8 });

    IScaleLayer* scale3bn = addBatchNorm2d(network, weightMap, *scale3pool->getOutput(0), lname + "scale3.1", 1e-5);

    IActivationLayer* scale3relu = network->addActivation(*scale3bn->getOutput(0), ActivationType::kRELU);
    IConvolutionLayer* scale3conv = network->addConvolution(*scale3relu->getOutput(0), branch_planes , DimsHW{ 1, 1 }, weightMap[lname + "scale3.3.weight"], emptywts);
    assert(scale3conv);
    scale3conv->setPadding(DimsHW{ 0, 0 });
    float *deval3 = reinterpret_cast<float*>(malloc(sizeof(float) * branch_planes * 8 * 8));
    for (int i = 0; i < branch_planes * 8 * 8; i++) {
        deval3[i] = 1.0;
    }
    Weights deconvwts3{ DataType::kFLOAT, deval3, branch_planes * 8 * 8 };
    IDeconvolutionLayer* scale3_interpolate = network->addDeconvolutionNd(*scale3conv->getOutput(0), branch_planes, DimsHW{ 8, 8 }, deconvwts3, emptywts);
    scale3_interpolate->setStrideNd(DimsHW{ 8, 8 });
    scale3_interpolate->setNbGroups(branch_planes);

    IElementWiseLayer* process3_input = network->addElementWise(*scale3_interpolate->getOutput(0), *process2conv->getOutput(0), ElementWiseOperation::kSUM);
// process3
    IScaleLayer* process3bn = addBatchNorm2d(network, weightMap, *process3_input->getOutput(0), lname + "process3.0", 1e-5);

    IActivationLayer* process3relu = network->addActivation(*process3bn->getOutput(0), ActivationType::kRELU);

    IConvolutionLayer* process3conv = network->addConvolution(*process3relu->getOutput(0), branch_planes , DimsHW{ 3, 3 }, weightMap[lname + "process3.2.weight"], emptywts);
    assert(process3conv);
    process3conv->setPadding(DimsHW{ 1, 1 });

//  scale4
    int input_w = input.getDimensions().d[3];
    int input_h = input.getDimensions().d[2];
    IPoolingLayer* scale4pool = network->addPooling(input, PoolingType::kAVERAGE, DimsHW{ input_h, input_w });
    assert(scale4pool);
    scale4pool->setStride(DimsHW{ input_h, input_w });
    scale4pool->setPadding(DimsHW{ 0, 0 });

    IScaleLayer* scale4bn = addBatchNorm2d(network, weightMap, *scale4pool->getOutput(0), lname + "scale4.1", 1e-5);

    IActivationLayer* scale4relu = network->addActivation(*scale4bn->getOutput(0), ActivationType::kRELU);
    IConvolutionLayer* scale4conv = network->addConvolution(*scale4relu->getOutput(0), branch_planes , DimsHW{ 1, 1 }, weightMap[lname + "scale4.3.weight"], emptywts);
    assert(scale4conv);
    scale4conv->setPadding(DimsHW{ 0, 0 });

    float *deval4 = reinterpret_cast<float*>(malloc(sizeof(float) * branch_planes * input_h * input_w));
    for (int i = 0; i < branch_planes * input_h * input_w; i++) {
        deval4[i] = 1.0;
    }
    Weights deconvwts4{ DataType::kFLOAT, deval4, branch_planes * input_h * input_w };
    IDeconvolutionLayer* scale4_interpolate = network->addDeconvolutionNd(*scale4conv->getOutput(0), branch_planes, DimsHW{ input_h, input_w }, deconvwts4, emptywts);
    scale4_interpolate->setStrideNd(DimsHW{ input_h, input_w });
    scale4_interpolate->setNbGroups(branch_planes);

    IElementWiseLayer* process4_input = network->addElementWise(*scale4_interpolate->getOutput(0), *process3conv->getOutput(0), ElementWiseOperation::kSUM);
// process4
    IScaleLayer* process4bn = addBatchNorm2d(network, weightMap, *process4_input->getOutput(0), lname + "process4.0", 1e-5);

    IActivationLayer* process4relu = network->addActivation(*process4bn->getOutput(0), ActivationType::kRELU);

    IConvolutionLayer* process4conv = network->addConvolution(*process4relu->getOutput(0), branch_planes , DimsHW{ 3, 3 }, weightMap[lname + "process4.2.weight"], emptywts);
    assert(process4conv);
    process4conv->setPadding(DimsHW{ 1, 1 });

//  compression
    // out = self.compression(torch.cat(x_list, 1)) + self.shortcut(x)
    ITensor* inputTensors[] = {scale0conv->getOutput(0),  process1conv->getOutput(0) ,  process2conv->getOutput(0), process3conv->getOutput(0), process4conv->getOutput(0)};
    IConcatenationLayer* neck_cat = network->addConcatenation(inputTensors, 5);

    IScaleLayer* compressionbn = addBatchNorm2d(network, weightMap, *neck_cat->getOutput(0), lname + "compression.0", 1e-5);

    IActivationLayer* compressionrelu = network->addActivation(*compressionbn->getOutput(0), ActivationType::kRELU);

    IConvolutionLayer* compressionconv = network->addConvolution(*compressionrelu->getOutput(0), outplanes , DimsHW{ 1, 1 }, weightMap[lname + "compression.2.weight"], emptywts);
    assert(compressionconv);
    compressionconv->setPadding(DimsHW{ 0, 0 });

    // shortcut
    IScaleLayer* shortcutbn = addBatchNorm2d(network, weightMap, input, lname + "shortcut.0", 1e-5);

    IActivationLayer* shortcutrelu = network->addActivation(*shortcutbn->getOutput(0), ActivationType::kRELU);

    IConvolutionLayer* shortcutconv = network->addConvolution(*shortcutrelu->getOutput(0), outplanes , DimsHW{ 1, 1 }, weightMap[lname + "shortcut.2.weight"], emptywts);
    assert(shortcutconv);
    shortcutconv->setPadding(DimsHW{ 0, 0 });

    IElementWiseLayer* out = network->addElementWise(*compressionconv->getOutput(0), *shortcutconv->getOutput(0), ElementWiseOperation::kSUM);
    return out;
}

ILayer* segmenthead(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int interplanes, int outplanes, std::string lname) {
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, input, lname + "bn1", 1e-5);

    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);


    IConvolutionLayer* conv1 = network->addConvolution(*relu1->getOutput(0), interplanes , DimsHW{ 3, 3 }, weightMap[lname + "conv1.weight"], emptywts);
    assert(conv1);
    conv1->setPadding(DimsHW{ 1, 1 });

    IScaleLayer* bn2 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "bn2", 1e-5);

    IActivationLayer* relu2 = network->addActivation(*bn2->getOutput(0), ActivationType::kRELU);
    IConvolutionLayer* conv2 = network->addConvolution(*relu2->getOutput(0), outplanes , DimsHW{ 1, 1 }, weightMap[lname + "conv2.weight"], weightMap[lname + "conv2.bias"]);
    assert(conv2);
    conv2->setPadding(DimsHW{ 0, 0 });

    return conv2;
}

int read_files_in_dir(const char *p_dir_name, std::vector<std::string> &file_names) {
    DIR *p_dir = opendir(p_dir_name);
    if (p_dir == nullptr) {
        return -1;
    }

    struct dirent* p_file = nullptr;
    while ((p_file = readdir(p_dir)) != nullptr) {
        if (strcmp(p_file->d_name, ".") != 0 &&
            strcmp(p_file->d_name, "..") != 0) {
            //std::string cur_file_name(p_dir_name);
            //cur_file_name += "/";
            //cur_file_name += p_file->d_name;
            std::string cur_file_name(p_file->d_name);
            file_names.push_back(cur_file_name);
        }
    }
    closedir(p_dir);
    return 0;
}

static const int map_[19][3] = { {255,0,0} ,
                                {128,0,0},
                                {0,128,0},
                                {0,0,128},
                                {128,128,0},
                                {128,0,128},
                                {0,128,128},
                                {0,255,0},
                                {0,0,255},
                                {255,0,0},
                                {255,255,0},
                                {0,255,255},
                                {255,0,255},
                                {255,0,128},
                                {128,255,0},
                                {128,0,255},
                                {0,255,128},
                                {0,255,255},
                                {255,0,255},};

cv::Mat map2cityscape(cv::Mat real_out,cv::Mat real_out_)
{
    for (int i = 0; i < 128; ++i)
    {
        cv::Vec<float, 19> *p1 = real_out.ptr<cv::Vec<float, 19>>(i);
        cv::Vec3b *p2 = real_out_.ptr<cv::Vec3b>(i);
        for (int j = 0; j < 128; ++j)
        {
            int index = 0;
            float swap;
            for (int c = 0; c < 19; ++c)
            {
                if (p1[j][0] < p1[j][c])
                {
                    swap = p1[j][0];
                    p1[j][0] = p1[j][c];
                    p1[j][c] = swap;
                    index = c;
                }
            }
            p2[j][0] = map_[index][2];
            p2[j][1] = map_[index][1];
            p2[j][2] = map_[index][0];

        }
    }
    return real_out_;
}

cv::Mat read2mat(float * prob, cv::Mat out)
{
    for (int i = 0; i < 128; ++i)
    {
        cv::Vec<float, 19> *p1 = out.ptr<cv::Vec<float, 19>>(i);
        for (int j = 0; j < 128; ++j)
        {
            for (int c = 0; c < 19; ++c)
            {
                p1[j][c] = prob[c * 128 * 128 + i * 128 + j];
            }
        }
    }
    return out;
}


#endif

