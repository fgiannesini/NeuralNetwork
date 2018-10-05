package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.container;

import com.fgiannesini.neuralnetwork.computer.data.LayerTypeData;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;

public class ForwardComputationContainer {

    private final LayerTypeData inputMatrix;
    private final NeuralNetworkModel neuralNetworkModel;

    public ForwardComputationContainer(LayerTypeData inputData, NeuralNetworkModel neuralNetworkModel) {
        this.inputMatrix = inputData;
        this.neuralNetworkModel = neuralNetworkModel;
    }

    public LayerTypeData getInput() {
        return inputMatrix;
    }

    public NeuralNetworkModel getNeuralNetworkModel() {
        return neuralNetworkModel;
    }
}
