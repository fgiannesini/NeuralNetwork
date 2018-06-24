package com.fgiannesini.neuralnetwork.computer;

import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;

public class OutputComputerBuilder {

    private NeuralNetworkModel neuralNetworkModel;

    private OutputComputerBuilder() {
    }

    static OutputComputerBuilder init() {
        return new OutputComputerBuilder();
    }

    public OutputComputerBuilder withModel(NeuralNetworkModel neuralNetworkModel) {
        this.neuralNetworkModel = neuralNetworkModel;
        return this;
    }

    public OutputComputer build() {
        return new OutputComputer(neuralNetworkModel);
    }

}
