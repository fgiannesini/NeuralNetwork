package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent;

import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import org.jblas.DoubleMatrix;

public class ForwardComputationContainer {

    private final DoubleMatrix inputMatrix;
    private final NeuralNetworkModel neuralNetworkModel;

    public ForwardComputationContainer(DoubleMatrix inputMatrix, NeuralNetworkModel neuralNetworkModel) {
        this.inputMatrix = inputMatrix;
        this.neuralNetworkModel = neuralNetworkModel;
    }

    public DoubleMatrix getInputMatrix() {
        return inputMatrix;
    }

    public NeuralNetworkModel getNeuralNetworkModel() {
        return neuralNetworkModel;
    }
}
