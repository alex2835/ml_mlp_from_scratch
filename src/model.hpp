#pragma once
#include <concepts>
#include <memory>
#include <ranges>
#include <vector>
#include "layer.hpp"

struct Model
{
    virtual ~Model() = default;
    virtual Mat forward(const Mat &x) = 0;
    virtual void backward(const Mat &dY) = 0;
    virtual void step(float lr, float l2 = 0.0f) = 0;
    virtual const Layer &layer(size_t i) const = 0;
    virtual size_t layer_count() const = 0;
};

// A plain chain of layers: output of one is the input of the next.
class Sequential : public Model
{
    std::vector<std::unique_ptr<Layer>> layers;

public:
    template <std::derived_from<Layer>... Ls>
    explicit Sequential(Ls... ls)
    {
        (layers.push_back(std::make_unique<Ls>(std::move(ls))), ...);
    }

    Mat forward(const Mat &x) override
    {
        Mat out = x;
        for (auto &layer : layers)
            out = layer->forward(out);
        return out;
    }

    void backward(const Mat &dY) override
    {
        Mat dL = dY;
        for (auto &layer : layers | std::views::reverse)
            dL = layer->backward(dL);
    }

    void step(float lr, float l2 = 0.0f) override
    {
        for (auto &layer : layers)
            layer->step(lr, l2);
    }

    const Layer &layer(size_t i) const override { return *layers[i]; }
    size_t layer_count() const override { return layers.size(); }
};
