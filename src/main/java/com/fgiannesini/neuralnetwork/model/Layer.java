package com.fgiannesini.neuralnetwork.model;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import com.fgiannesini.neuralnetwork.initializer.Initializer;
import org.jblas.DoubleMatrix;

import java.util.List;
import java.util.Objects;

public abstract class Layer implements Cloneable {
    private final int inputLayerSize;
    private final int outputLayerSize;
    private final ActivationFunctionType activationFunctionType;
    private DoubleMatrix weightMatrix;

    Layer(int inputLayerSize, int outputLayerSize, ActivationFunctionType activationFunctionType, Initializer initializer) {
        this.inputLayerSize = inputLayerSize;
        this.activationFunctionType = activationFunctionType;
        weightMatrix = initializer.initDoubleMatrix(outputLayerSize, inputLayerSize);
        this.outputLayerSize = outputLayerSize;
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

    public void setWeightMatrix(DoubleMatrix weightMatrix) {
        this.weightMatrix = weightMatrix;
    }

    public ActivationFunctionType getActivationFunctionType() {
        return activationFunctionType;
    }

    public abstract List<DoubleMatrix> getParametersMatrix();

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Layer layer = (Layer) o;
        return inputLayerSize == layer.inputLayerSize &&
                outputLayerSize == layer.outputLayerSize &&
                activationFunctionType == layer.activationFunctionType &&
                Objects.equals(weightMatrix, layer.weightMatrix);
    }

    @Override
    public int hashCode() {
        return Objects.hash(inputLayerSize, outputLayerSize, activationFunctionType, weightMatrix);
    }

    @Override
    public Layer clone() {
        try {
            Layer clone = (Layer) super.clone();
            clone.weightMatrix = weightMatrix.dup();
            return clone;
        } catch (CloneNotSupportedException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public String toString() {
        return "Layer{" +
                "weightMatrix=" + weightMatrix +
                '}';
    }
}
