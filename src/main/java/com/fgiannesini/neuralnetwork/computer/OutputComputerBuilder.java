package com.fgiannesini.neuralnetwork.computer;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;

public class OutputComputerBuilder {

    private NeuralNetworkModel neuralNetworkModel;
    private ActivationFunctionType activationFunctionType;

    private OutputComputerBuilder() {
        activationFunctionType = ActivationFunctionType.NONE;
    }

    static OutputComputerBuilder init() {
        return new OutputComputerBuilder();
    }

    public OutputComputerBuilder withModel(NeuralNetworkModel neuralNetworkModel) {
        this.neuralNetworkModel = neuralNetworkModel;
        return this;
    }

    public OutputComputerBuilder withActivationFunction(ActivationFunctionType activationFunctionType) {
        this.activationFunctionType = activationFunctionType;
        return this;
    }

    public OutputComputer build() {
        return new OutputComputer(neuralNetworkModel, activationFunctionType.getActivationFunction());
    }

}
