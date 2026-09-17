#pragma once
#include "helpers.hpp"

// Every layer maps a batch [in x B] to [out x B]; each column is one sample.
//
// Conv/pool layers interpret a column as an image tensor flattened in HWC
// order: index = (y*W + x)*C + c (pixel-major, channel fastest). For C = 1
// this is exactly the row-major 28x28 MNIST layout, and because a column is
// already flat, no separate Flatten layer is ever needed before a Dense.
struct Layer
{
    virtual ~Layer() = default;

    virtual Mat forward(const Mat &x) = 0;
    virtual Mat backward(const Mat &dY) = 0;     // returns dL/dX for the layer below
    virtual void step(float /*lr*/, float /*l2*/) {} // parameter-free layers do nothing

    // Trainable weight matrix, or an empty matrix for parameter-free layers.
    virtual const Mat &weights() const
    {
        static const Mat none;
        return none;
    }
};
