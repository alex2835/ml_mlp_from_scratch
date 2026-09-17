#include <iostream>
#include "conv_im2col.hpp"
#include "conv_naive.hpp"
#include "dense_layer.hpp"
#include "maxpool.hpp"
#include "train.hpp"
#include "visualize.hpp"
#include "mnist/mnist_reader.hpp"

// Pack images into a [784 x N] matrix, one image per column, pixels scaled to [0, 1].
inline Mat to_matrix(const std::vector<std::vector<uint8_t>> &images)
{
    Mat X(images[0].size(), images.size());
    for (size_t i = 0; i < images.size(); ++i)
        for (size_t p = 0; p < images[i].size(); ++p)
            X(p, i) = images[i][p] / 255.0f;
    return X;
}

// Labels -> one-hot matrix [10 x N].
inline Mat to_one_hot(const std::vector<uint8_t> &labels, int classes)
{
    Mat Y = Mat::Zero(classes, labels.size());
    for (size_t i = 0; i < labels.size(); ++i)
        Y(labels[i], i) = 1.0f;
    return Y;
}

// The two conv implementations must agree exactly (same weights, same input,
// same incoming gradient) — otherwise one of them has an indexing bug.
void check_conv_equivalence()
{
    ConvNaive  naive(2, 6, 6, 3, 3, 1, Activation::ReLU);
    ConvIm2col fast (2, 6, 6, 3, 3, 1, Activation::ReLU);
    fast.K = naive.K;
    fast.b = naive.b;

    Mat x  = Mat::Random(6 * 6 * 2, 4);
    Mat y1 = naive.forward(x), y2 = fast.forward(x);
    Mat dY = Mat::Random(y1.rows(), y1.cols());
    Mat dx1 = naive.backward(dY), dx2 = fast.backward(dY);

    std::cout << "conv naive vs im2col, max abs diff:"
              << "  Y " << (y1 - y2).cwiseAbs().maxCoeff()
              << "  dX " << (dx1 - dx2).cwiseAbs().maxCoeff()
              << "  dK " << (naive.dK - fast.dK).cwiseAbs().maxCoeff()
              << "  db " << (naive.db - fast.db).cwiseAbs().maxCoeff() << "\n\n";
}

int main()
{
    std::cout.setf(std::ios::unitbuf); // flush every write so logs show live when redirected
    check_conv_equivalence();

    auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_DATA_LOCATION);

    Mat X_train = to_matrix(dataset.training_images); // [784 x 60000]
    Mat Y_train = to_one_hot(dataset.training_labels, 10);
    Mat X_test  = to_matrix(dataset.test_images);     // [784 x 10000]

    // LeNet-style stack. Shapes per column: 28x28x1 -> 28x28x8 -> 14x14x8
    // -> 14x14x16 -> 7x7x16 = 784 -> 128 -> 10.
    auto make_cnn = [](auto make_conv) {
        return std::make_unique<Sequential>(
            make_conv(1, 28, 28, 8, 3, 1, Activation::ReLU),
            MaxPool2D(8, 28, 28),
            make_conv(8, 14, 14, 16, 3, 1, Activation::ReLU),
            MaxPool2D(16, 14, 14),
            DenseLayer(7 * 7 * 16, 128, Activation::ReLU),
            DenseLayer(128, 10, Activation::Linear));
    };
    auto naive_conv  = [](auto... a) { return ConvNaive(a...); };
    auto im2col_conv = [](auto... a) { return ConvIm2col(a...); };

    TrainConfig experiments[] = {
        {.name       = "linear",
         .make_model = [] { return std::make_unique<Sequential>(
                                DenseLayer(784, 10, Activation::Linear)); }},

        {.name       = "mlp_relu_128",
         .make_model = [] { return std::make_unique<Sequential>(
                                DenseLayer(784, 128, Activation::ReLU),
                                DenseLayer(128, 10, Activation::Linear)); }},

        // reference implementation: too slow for the full set, so a small slice
        {.name          = "cnn_naive",
         .make_model    = [&] { return make_cnn(naive_conv); },
         .epochs        = 1,
         .train_samples = 2560,
         .test_samples  = 1000},

        {.name       = "cnn_im2col",
         .make_model = [&] { return make_cnn(im2col_conv); },
         .epochs     = 5},
    };

    for (const TrainConfig &cfg : experiments)
    {
        std::unique_ptr<Model> model = train(X_train, Y_train, X_test, dataset.test_labels, cfg);

        // only a first layer with 784-wide weight rows reshapes to 28x28 pictures
        const Mat &W = model->layer(0).weights();
        if (W.cols() == 28 * 28)
        {
            std::string dir = "pictures/" + cfg.name;
            save_weight_images(W, 28, 28, dir);
            std::cout << "saved weight images to " << dir << "/\n";
        }
        std::cout << "\n";
    }

    return 0;
}
