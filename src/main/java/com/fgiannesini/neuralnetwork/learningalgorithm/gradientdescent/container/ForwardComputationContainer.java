package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.container;

import com.fgiannesini.neuralnetwork.model.Layer;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import org.jblas.DoubleMatrix;

public class ForwardComputationContainer {

    private final DoubleMatrix inputMatrix;
    private final NeuralNetworkModel<Layer> neuralNetworkModel;

    public ForwardComputationContainer(DoubleMatrix inputMatrix, NeuralNetworkModel<Layer> neuralNetworkModel) {
        this.inputMatrix = inputMatrix;
        this.neuralNetworkModel = neuralNetworkModel;
    }

    public DoubleMatrix getInputMatrix() {
        return inputMatrix;
    }

    public NeuralNetworkModel<Layer> getNeuralNetworkModel() {
        return neuralNetworkModel;
    }
}
