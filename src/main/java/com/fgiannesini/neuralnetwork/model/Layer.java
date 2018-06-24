package com.fgiannesini.neuralnetwork.model;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import com.fgiannesini.neuralnetwork.initializer.Initializer;
import org.jblas.FloatMatrix;

public class Layer {

    private int inputLayerSize;
    private int outputLayerSize;
    private ActivationFunctionType activationFunctionType;
    private FloatMatrix weightMatrix;
    private FloatMatrix biasMatrix;

    public Layer(int inputLayerSize, int outputLayerSize, Initializer initializer, ActivationFunctionType activationFunctionType) {
        this.inputLayerSize = inputLayerSize;
        this.outputLayerSize = outputLayerSize;
        this.activationFunctionType = activationFunctionType;
        weightMatrix = initializer.initFloatMatrix(inputLayerSize, outputLayerSize);
        biasMatrix = initializer.initFloatMatrix(1, outputLayerSize);
    }

    public int getInputLayerSize() {
        return inputLayerSize;
    }

    public int getOutputLayerSize() {
        return outputLayerSize;
    }

    public FloatMatrix getWeightMatrix() {
        return weightMatrix;
    }

    public FloatMatrix getBiasMatrix() {
        return biasMatrix;
    }

    public ActivationFunctionType getActivationFunctionType() {
        return activationFunctionType;
    }
}
