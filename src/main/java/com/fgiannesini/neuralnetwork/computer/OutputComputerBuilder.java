package com.fgiannesini.neuralnetwork.computer;

import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;

public class OutputComputerBuilder {

    private NeuralNetworkModel neuralNetworkModel;

    private OutputComputerBuilder() {
    }

    public static OutputComputerBuilder init() {
        return new OutputComputerBuilder();
    }

    public OutputComputerBuilder withModel(NeuralNetworkModel neuralNetworkModel) {
        this.neuralNetworkModel = neuralNetworkModel;
        return this;
    }

    public FinalOutputComputer buildFinalOutputComputer() {
        checkParameters();
        return new FinalOutputComputer(neuralNetworkModel);
    }

    public IntermediateOutputComputer buildIntermediateOutputComputer() {
        checkParameters();
        return new IntermediateOutputComputer(neuralNetworkModel);
    }

    private void checkParameters() {
        if (neuralNetworkModel == null) {
            throw new IllegalArgumentException("Missing neural Network");
        }
    }
}
