package com.fgiannesini.neuralnetwork.gradient;

import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;

public class GradientDescentBuilder {

    private NeuralNetworkModel neuralNetworkModel;

    private GradientDescentBuilder() {
    }

    public static GradientDescentBuilder init() {
        return new GradientDescentBuilder();
    }

    public GradientDescentBuilder withModel(NeuralNetworkModel neuralNetworkModel) {
        this.neuralNetworkModel = neuralNetworkModel;
        return this;
    }

    public GradientDescent build() {
        if (neuralNetworkModel == null) {
            throw new IllegalArgumentException("NeuralNetworkModel missing");
        }
        return new GradientDescent(neuralNetworkModel, 0.01f);
    }

}
