package com.fgiannesini.neuralnetwork.learningalgorithm;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import org.jblas.DoubleMatrix;

class GradientDescentLayerResult {
    private DoubleMatrix aLayerResults;
    private final DoubleMatrix weightMatrix;

    private ActivationFunctionType activationFunctionType;

    public GradientDescentLayerResult() {
        this.weightMatrix = DoubleMatrix.EMPTY;
    }

    public GradientDescentLayerResult(DoubleMatrix weightMatrix, ActivationFunctionType activationFunctionType) {
        this.weightMatrix = weightMatrix.dup();
        this.activationFunctionType = activationFunctionType;
    }

    public ActivationFunctionType getActivationFunctionType() {
        return activationFunctionType;
    }

    public DoubleMatrix getWeightMatrix() {
        return weightMatrix;
    }

    void setAResultLayer(DoubleMatrix aLayerResult) {
        aLayerResults = aLayerResult;
    }

    public DoubleMatrix getaLayerResults() {
        return aLayerResults;
    }

}
