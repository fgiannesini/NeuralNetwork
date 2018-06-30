package com.fgiannesini.neuralnetwork.model;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import com.fgiannesini.neuralnetwork.initializer.Initializer;
import org.jblas.DoubleMatrix;

import java.util.Objects;

public class Layer implements Cloneable {

    private final int inputLayerSize;
    private final int outputLayerSize;
    private final ActivationFunctionType activationFunctionType;
    private DoubleMatrix weightMatrix;
    private DoubleMatrix biasMatrix;

    public Layer(int inputLayerSize, int outputLayerSize, Initializer initializer, ActivationFunctionType activationFunctionType) {
        this.inputLayerSize = inputLayerSize;
        this.outputLayerSize = outputLayerSize;
        this.activationFunctionType = activationFunctionType;
        weightMatrix = initializer.initDoubleMatrix(inputLayerSize, outputLayerSize);
        biasMatrix = initializer.initDoubleMatrix(1, outputLayerSize);
    }

    public int getInputLayerSize() {
        return inputLayerSize;
    }

    public int getOutputLayerSize() {
        return outputLayerSize;
    }

    public DoubleMatrix getWeightMatrix() {
        return weightMatrix;
    }

    public DoubleMatrix getBiasMatrix() {
        return biasMatrix;
    }

    public ActivationFunctionType getActivationFunctionType() {
        return activationFunctionType;
    }

    @Override
    public Layer clone() {
        try {
            Layer clone = (Layer) super.clone();
            clone.biasMatrix = biasMatrix.dup();
            clone.weightMatrix = weightMatrix.dup();
            return clone;
        } catch (CloneNotSupportedException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof Layer)) return false;
        Layer layer = (Layer) o;
        return inputLayerSize == layer.inputLayerSize &&
                outputLayerSize == layer.outputLayerSize &&
                activationFunctionType == layer.activationFunctionType &&
                Objects.equals(weightMatrix, layer.weightMatrix) &&
                Objects.equals(biasMatrix, layer.biasMatrix);
    }

    @Override
    public int hashCode() {
        return Objects.hash(inputLayerSize, outputLayerSize, activationFunctionType, weightMatrix, biasMatrix);
    }
}
