#include "YoloV5Common.h"
#include "BoxReslutToJson.h"
#include "cuda_utils.h"
#include "YoloPreProccessCUDA.h"

#define NMS_THRESH 0.4
#define CONF_THRESH 0.5


inline int my_softmax(float* input){
        float fm = 0;
        for(int i = 0;i < 3;i++){
            fm += std::exp(input[i]);
        }
        float x1 = std::exp(input[0]) / fm;
        float x2 = std::exp(input[1]) / fm;
        float x3 = std::exp(input[2]) / fm;
        return std::max(x3,std::max(x1,x2));
}

void preprocess_image(cv::Mat& img, std::vector<uint8_t>& data) {
    int kClsInputW = 224;
    int kClsInputH = 224;
    cv::resize(img, img, cv::Size(kClsInputW, kClsInputH));
    int num_elements = 3 * img.rows * img.cols;
    data.clear();
    for(int row = 0; row < img.rows; ++row) {
        uint8_t* uc_pixel = img.data + row * img.step;
        for (int col = 0; col < img.cols; ++col) {
            uint8_t blue = static_cast<uint8_t>(((float)uc_pixel[0] / 255.0 - 0.406) / 0.225 * 255);
            uint8_t green = static_cast<uint8_t>(((float)uc_pixel[1] / 255.0 - 0.456) / 0.224 * 255);
            uint8_t red = static_cast<uint8_t>(((float)uc_pixel[2] / 255.0 - 0.485) / 0.229 * 255);

            data.push_back(red);
            data.push_back(green);
            data.push_back(blue);

            uc_pixel += 3; // 移动到下一个像素
        }
    }
}

class Coal_QdClsYolov8Infer{
public:
    Coal_QdClsYolov8Infer(std::unique_ptr<tc::InferenceServerGrpcClient> &&client, int height, int width,int targetHeight,int targetWidth,
                           std::unique_ptr<tc::InferOptions> options){
//        this->preProcess = std::move(preProcess);
        this->client = std::move(client);
        this->options = std::move(options);
        this->INPUT_C = 3;
        this->INPUT_H = height;
        this->INPUT_W = width;
        this->targetHeight=targetHeight;
        this->targetWidth=targetWidth;
        std::vector<int64_t> inputShape{3, targetHeight, targetWidth};
        tc::InferInput * input;
        auto msg = tc::InferInput::Create(&input,INPUT_DATA_NAME,inputShape,DATA_TYPE);
        if(!msg.IsOk()) spdlog::error(msg.Message());
        inputPtr.reset(input);
        tc::InferRequestedOutput* output;
        msg = tc::InferRequestedOutput::Create(&output,OUTPUT_DATA_NAME);
        if(!msg.IsOk()) spdlog::error(msg.Message());
        outputPtr.reset(output);
    }
    static std::unique_ptr<Coal_QdClsYolov8Infer>
    CreateCoal_QdClsYolov8Infer(const std::string &&serverURL, const int Height, const int Width){
        std::unique_ptr<tc::InferenceServerGrpcClient> client;
        // 创建客户端
        auto msg = tc::InferenceServerGrpcClient::Create(&client, serverURL);
        auto option = std::make_unique<tc::InferOptions>("yolov8s-QD");
        option->model_version_ = "1";
//        auto preProcess = YoloPreProcessCUDA::createUnfixedImagePreProcessCUDA(224, 224);
        if (!msg.IsOk()) {
            //如果创建失败返回空指针
            spdlog::error("unable to create triton client {}", msg.Message());
        }
        auto res = new Coal_QdClsYolov8Infer(std::move(client), Height,Width,224,
                                             224,
                                             std::move(option));
        return std::unique_ptr<Coal_QdClsYolov8Infer>(res);
    }
    json infer(cv::Mat &Mat) {
//        auto data = preProcess->preProcess(Mat);
        std::vector<uint8_t> data;
        preprocess_image(Mat, data);
        //充值图像输入
        inputPtr->Reset();
        //填充图像数据
        auto msg = inputPtr->AppendRaw(data);
        if (!msg.IsOk()) {
            spdlog::error("request error {}", msg.Message());
        }
        tc::InferResult *result;
        msg = client->Infer(&result, *options, {inputPtr.get()});
        if (!msg.IsOk()) {
            spdlog::error(msg.Message());
        }

        // Get pointers to the result returned...
        float *output_data;
        size_t output_byte_size;
        std::shared_ptr<tc::InferResult> resultPtr;
        resultPtr.reset(result);

        resultPtr->RawData("prob", (const uint8_t **) &output_data, &output_byte_size);
        std::cout << *output_data  << " " << *(output_data+1) << " " <<  *(output_data+2) << std::endl;
        ////将检测结果保存并返回
        json QD_ClsResult;
        // 假设output_data是一个浮点数组，其中包含了对每个类别的预测置信度
        // 找到最大的置信度的索引，即为分类结果
        my_softmax(output_data);
        int classID = std::distance((float*)output_data, std::max_element(output_data, output_data + 3));
        float confidence = output_data[classID];
        QD_ClsResult["classID"] = classID;
        QD_ClsResult["confidence"] = confidence;

        // 创建一个带有分类结果的文本字符串
        std::string result_text = "Class: " + std::to_string(classID) + ", Confidence: " + std::to_string(confidence);

        // 在图像上绘制分类结果
        cv::putText(Mat, result_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);

        // 保存带有分类结果的图像
        cv::imwrite("result_image.jpg", Mat);

        spdlog::info("QD infer complete!");

        return QD_ClsResult;
    }
private:
    std::unique_ptr<tc::InferenceServerGrpcClient> client;
    cudaStream_t stream;
    uint8_t* img_host = nullptr;
    uint8_t* img_device = nullptr;
    uint8_t* dst_device= nullptr;
    uint8_t* dst_host= nullptr;
    int64_t INPUT_H;
//输入数据的名称
    const std::string INPUT_DATA_NAME="data";
//输出数据的名称
    const std::string OUTPUT_DATA_NAME="data";
//输入数据的类型
    const std::string DATA_TYPE="FP32" ;
private:
    int64_t INPUT_W;
    int64_t INPUT_C;
    int64_t INPUT_BS;
    int targetHeight;
    int targetWidth;
    const int MAX_IMAGE_INPUT_SIZE_THRESH=3000*3000;
    std::shared_ptr<tc::InferInput> inputPtr;
    std::shared_ptr<tc::InferRequestedOutput> outputPtr;
    std::unique_ptr<tc::InferOptions> options;
    std::unique_ptr<YoloPreProcessCUDA> preProcess;

    };