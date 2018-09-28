package com.fgiannesini.neuralnetwork.model;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import com.fgiannesini.neuralnetwork.initializer.Initializer;
import org.jblas.DoubleMatrix;

import java.util.Arrays;
import java.util.List;
import java.util.Objects;

public class WeightBiasLayer extends Layer implements Cloneable {

    private int inputLayerSize;
    private int outputLayerSize;
    private DoubleMatrix weightMatrix;
    private DoubleMatrix biasMatrix;

    public WeightBiasLayer(int inputLayerSize, int outputLayerSize, Initializer initializer, ActivationFunctionType activationFunctionType) {
        super(activationFunctionType);
        weightMatrix = initializer.initDoubleMatrix(outputLayerSize, inputLayerSize);
        biasMatrix = initializer.initDoubleMatrix(outputLayerSize, 1);
        this.inputLayerSize = inputLayerSize;
        this.outputLayerSize = outputLayerSize;
    }

    public DoubleMatrix getBiasMatrix() {
        return biasMatrix;
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

    public void setBiasMatrix(DoubleMatrix biasMatrix) {
        this.biasMatrix = biasMatrix;
    }

    @Override
    public WeightBiasLayer clone() {
        WeightBiasLayer clone = (WeightBiasLayer) super.clone();
        clone.weightMatrix = weightMatrix.dup();
        clone.biasMatrix = biasMatrix.dup();
        return clone;
    }

    @Override
    public List<DoubleMatrix> getParametersMatrix() {
        return Arrays.asList(getWeightMatrix(), getBiasMatrix());
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        WeightBiasLayer that = (WeightBiasLayer) o;
        return inputLayerSize == that.inputLayerSize &&
                outputLayerSize == that.outputLayerSize &&
                Objects.equals(weightMatrix, that.weightMatrix) &&
                Objects.equals(biasMatrix, that.biasMatrix);
    }

    @Override
    public int hashCode() {
        return Objects.hash(inputLayerSize, outputLayerSize, weightMatrix, biasMatrix);
    }

    @Override
    public String toString() {
        return "WeightBiasLayer{" +
                "inputLayerSize=" + inputLayerSize +
                ", outputLayerSize=" + outputLayerSize +
                ", weightMatrix=" + weightMatrix +
                ", biasMatrix=" + biasMatrix +
                '}';
    }

    @Override
    public void accept(LayerVisitor layerVisitor) {
        layerVisitor.visit(this);
    }
}
