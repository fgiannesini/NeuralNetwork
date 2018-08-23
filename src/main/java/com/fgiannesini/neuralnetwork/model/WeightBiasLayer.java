package com.fgiannesini.neuralnetwork.model;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import com.fgiannesini.neuralnetwork.initializer.Initializer;
import org.jblas.DoubleMatrix;

import java.util.Objects;

public class WeightBiasLayer extends Layer implements Cloneable {

    private DoubleMatrix biasMatrix;

    public WeightBiasLayer(int inputLayerSize, int outputLayerSize, Initializer initializer, ActivationFunctionType activationFunctionType) {
        super(inputLayerSize, outputLayerSize, activationFunctionType, initializer);
        biasMatrix = initializer.initDoubleMatrix(outputLayerSize, 1);
    }

    public DoubleMatrix getBiasMatrix() {
        return biasMatrix;
    }

    public void setBiasMatrix(DoubleMatrix biasMatrix) {
        this.biasMatrix = biasMatrix;
    }

    @Override
    public WeightBiasLayer clone() {
        WeightBiasLayer clone = (WeightBiasLayer) super.clone();
        clone.biasMatrix = biasMatrix.dup();
        return clone;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof WeightBiasLayer)) return false;
        if (!super.equals(o)) return false;
        WeightBiasLayer that = (WeightBiasLayer) o;
        return Objects.equals(biasMatrix, that.biasMatrix);
    }

    @Override
    public int hashCode() {
        return Objects.hash(super.hashCode(), biasMatrix);
    }

    @Override
    public String toString() {
        return "WeightBiasLayer{" +
                "biasMatrix=" + biasMatrix +
                "} " + super.toString();
    }
}
