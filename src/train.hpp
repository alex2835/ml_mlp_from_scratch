#pragma once
#include <chrono>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include "model.hpp"

struct TrainConfig
{
    std::string name;
    // Builds a fresh model. The last layer must be Activation::Linear because
    // the train loop folds softmax's derivative into dZ = P - Y itself.
    std::function<std::unique_ptr<Model>()> make_model;
    int epochs = 10;
    int batch_size = 128;
    float lr = 0.1f;
    float l2 = 0.0f;       // L2 / weight-decay strength, 0 = off
    int train_samples = 0; // 0 = whole set; cap it for slow reference models
    int test_samples = 0;
};

// Forward a big matrix in chunks so conv layers don't allocate im2col
// buffers for 10 000 images at once.
inline Mat predict(Model &model, const Mat &X, int chunk = 500)
{
    Mat out;
    for (int start = 0; start < X.cols(); start += chunk)
    {
        int n = std::min(chunk, int(X.cols()) - start);
        Mat part = model.forward(X.middleCols(start, n));
        if (out.size() == 0) out.resize(part.rows(), X.cols());
        out.middleCols(start, n) = part;
    }
    return out;
}

// Mini-batch SGD on softmax + cross-entropy. Prints progress per epoch
// and returns the trained model.
inline std::unique_ptr<Model> train(const Mat &X_train_full,
                                    const Mat &Y_train_full,
                                    const Mat &X_test_full,
                                    const std::vector<uint8_t> &test_labels_full,
                                    const TrainConfig &cfg)
{
    int n_train = cfg.train_samples ? cfg.train_samples : int(X_train_full.cols());
    int n_test  = cfg.test_samples  ? cfg.test_samples  : int(X_test_full.cols());
    Mat X_train = X_train_full.leftCols(n_train);
    Mat Y_train = Y_train_full.leftCols(n_train);
    Mat X_test  = X_test_full.leftCols(n_test);
    std::vector<uint8_t> test_labels(test_labels_full.begin(), test_labels_full.begin() + n_test);

    std::unique_ptr<Model> model = cfg.make_model();
    std::cout << "=== " << cfg.name << " (" << model->layer_count() << " layers, l2 = " << cfg.l2
              << ", train " << n_train << ", test " << n_test << ") ===\n";

    std::mt19937 rng(42);
    std::vector<int> order(n_train);
    std::iota(order.begin(), order.end(), 0);

    for (int epoch = 1; epoch <= cfg.epochs; ++epoch)
    {
        auto t0 = std::chrono::steady_clock::now();
        std::shuffle(order.begin(), order.end(), rng);

        float loss_sum = 0;
        int   batches  = 0;
        for (int start = 0; start + cfg.batch_size <= n_train; start += cfg.batch_size)
        {
            Mat X(X_train.rows(), cfg.batch_size);
            Mat Y(Y_train.rows(), cfg.batch_size);
            for (int i = 0; i < cfg.batch_size; ++i)
            {
                X.col(i) = X_train.col(order[start + i]);
                Y.col(i) = Y_train.col(order[start + i]);
            }

            Mat P = softmax(model->forward(X));
            loss_sum += cross_entropy(P, Y);
            ++batches;

            // For softmax + cross-entropy, dL/dZ = P - Y (averaged over the batch)
            model->backward((P - Y) / float(cfg.batch_size));
            model->step(cfg.lr, cfg.l2);
        }

        auto secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::cout << "epoch " << epoch
                  << "  train loss " << loss_sum / batches
                  << "  test accuracy " << accuracy(predict(*model, X_test), test_labels)
                  << "  (" << int(secs) << " s)\n";
    }

    return model;
}
