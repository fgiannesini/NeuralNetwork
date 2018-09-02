package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.container;

import com.fgiannesini.neuralnetwork.model.Layer;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import org.jblas.DoubleMatrix;

public class ForwardComputationContainer<L extends Layer> {

    private final DoubleMatrix inputMatrix;
    private final NeuralNetworkModel<L> neuralNetworkModel;

    public ForwardComputationContainer(DoubleMatrix inputMatrix, NeuralNetworkModel<L> neuralNetworkModel) {
        this.inputMatrix = inputMatrix;
        this.neuralNetworkModel = neuralNetworkModel;
    }

    public DoubleMatrix getInputMatrix() {
        return inputMatrix;
    }

    public NeuralNetworkModel<L> getNeuralNetworkModel() {
        return neuralNetworkModel;
    }
}
