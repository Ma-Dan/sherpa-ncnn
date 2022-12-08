#include <jni.h>
#include <thread>

#include "sherpa-ncnn/csrc/decode.h"
#include "sherpa-ncnn/csrc/features.h"
#include "sherpa-ncnn/csrc/lstm-model.h"
#include "sherpa-ncnn/csrc/symbol-table.h"

namespace sherpa {
    sherpa_ncnn::SymbolTable sym;
    std::unique_ptr<sherpa_ncnn::Model> model;
    sherpa_ncnn::FeatureExtractor feature_extractor;
    ncnn::Mat decoder_input;
    ncnn::Mat decoder_out;
    std::vector<int32_t> hyp;

    int32_t segment;
    int32_t offset;

    int32_t context_size;
    int32_t blank_id;

    int32_t num_tokens;
    int32_t num_processed;

    std::string result;

    bool stop = false;

    void init(JNIEnv *env, jobject, jstring jModelDir) {
        const char *pModelDir = env->GetStringUTFChars(jModelDir, nullptr);

        sherpa_ncnn::ModelConfig config;

        config.encoder_param = std::string(pModelDir) + "/encoder_jit_trace-pnnx.ncnn.param";
        config.encoder_bin = std::string(pModelDir) + "/encoder_jit_trace-pnnx.ncnn.bin";
        config.decoder_param = std::string(pModelDir) + "/decoder_jit_trace-pnnx.ncnn.param";
        config.decoder_bin = std::string(pModelDir) + "/decoder_jit_trace-pnnx.ncnn.bin";
        config.joiner_param = std::string(pModelDir) + "/joiner_jit_trace-pnnx.ncnn.param";
        config.joiner_bin = std::string(pModelDir) + "/joiner_jit_trace-pnnx.ncnn.bin";
        config.num_threads = 4;

        std::string tokens = std::string(pModelDir) + "/tokens.txt";
        sym = sherpa_ncnn::SymbolTable(tokens);

        model = sherpa_ncnn::Model::Create(config);

        segment = model->Segment();
        offset = model->Offset();

        context_size = model->ContextSize();
        blank_id = model->BlankId();

        hyp = std::vector<int32_t>(context_size, blank_id);

        decoder_input = ncnn::Mat(context_size);
        for (int32_t i = 0; i != context_size; ++i) {
            static_cast<int32_t *>(decoder_input)[i] = blank_id;
        }

        decoder_out = model->RunDecoder(decoder_input);

        num_tokens = hyp.size();
        num_processed = 0;
    }

    void reset(JNIEnv *env, jobject) {
        //TODO:重置LSTM
        hyp.clear();
        num_tokens = hyp.size();
        num_processed = 0;
        result = "";
    }

    void accept_waveform(JNIEnv *env, jobject, jfloatArray jWaveform) {
        jsize size = env->GetArrayLength(jWaveform);
        float_t* waveform = env->GetFloatArrayElements(jWaveform, 0);
        feature_extractor.AcceptWaveform(
                16000, reinterpret_cast<const float *>(waveform), size);
    }

    void decode_thread_func() {
        std::vector<ncnn::Mat> states;
        ncnn::Mat encoder_out;

        while (!stop) {
            while (feature_extractor.NumFramesReady() - num_processed >= segment) {
                ncnn::Mat features = feature_extractor.GetFrames(num_processed, segment);
                num_processed += offset;

                std::tie(encoder_out, states) = model->RunEncoder(features, states);

                sherpa_ncnn::GreedySearch(model.get(), encoder_out, &decoder_out, &hyp);

                if (hyp.size() != num_tokens) {
                    num_tokens = hyp.size();
                    std::string text;
                    for (int32_t i = context_size; i != hyp.size(); ++i) {
                        text += sym[hyp[i]];
                    }
                    fprintf(stderr, "%s\n", text.c_str());
                    result = text;
                }
            }
        }
    }

    void start_decode() {
        stop = false;
        std::thread decode_thread(decode_thread_func);
        decode_thread.detach();
    }

    void set_input_finished() {
        stop = true;
    }

    jstring get_result(JNIEnv *env, jobject) {
        return env->NewStringUTF((result).c_str());
    }
}

JNIEXPORT jint JNI_OnLoad(JavaVM *vm, void *) {
    JNIEnv *env;
    if (vm->GetEnv(reinterpret_cast<void **>(&env), JNI_VERSION_1_6) != JNI_OK) {
        return JNI_ERR;
    }

    jclass c = env->FindClass("com/k2fsa/sherpa/Recognize");
    if (c == nullptr) {
        return JNI_ERR;
    }

    static const JNINativeMethod methods[] = {
            {"init", "(Ljava/lang/String;)V", reinterpret_cast<void *>(sherpa::init)},
            {"reset", "()V", reinterpret_cast<void *>(sherpa::reset)},
            {"acceptWaveform", "([F)V",
                        reinterpret_cast<void *>(sherpa::accept_waveform)},
            {"setInputFinished", "()V",
                    reinterpret_cast<void *>(sherpa::set_input_finished)},
            {"startDecode", "()V", reinterpret_cast<void *>(sherpa::start_decode)},
            {"getResult", "()Ljava/lang/String;",
                                   reinterpret_cast<void *>(sherpa::get_result)}
    };
    int rc = env->RegisterNatives(c, methods,
                                  sizeof(methods) / sizeof(JNINativeMethod));

    if (rc != JNI_OK) {
        return rc;
    }

    return JNI_VERSION_1_6;
}

