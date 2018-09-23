package com.fgiannesini.neuralnetwork.model;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import com.fgiannesini.neuralnetwork.initializer.Initializer;
import org.jblas.DoubleMatrix;

import java.util.Arrays;
import java.util.List;
import java.util.Objects;

public class BatchNormLayer extends Layer implements Cloneable {

    private int inputLayerSize;
    private int outputLayerSize;
    private DoubleMatrix weightMatrix;
    private DoubleMatrix gammaMatrix;
    private DoubleMatrix betaMatrix;

    public BatchNormLayer(int inputLayerSize, int outputLayerSize, Initializer initializer, ActivationFunctionType activationFunctionType) {
        super(activationFunctionType);
        weightMatrix = initializer.initDoubleMatrix(outputLayerSize, inputLayerSize);
        gammaMatrix = initializer.initDoubleMatrix(outputLayerSize, 1);
        betaMatrix = initializer.initDoubleMatrix(outputLayerSize, 1);
        this.inputLayerSize = inputLayerSize;
        this.outputLayerSize = outputLayerSize;
    }

    public DoubleMatrix getGammaMatrix() {
        return gammaMatrix;
    }

    public DoubleMatrix getBetaMatrix() {
        return betaMatrix;
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

    @Override
    public BatchNormLayer clone() {
        BatchNormLayer clone = (BatchNormLayer) super.clone();
        clone.weightMatrix = weightMatrix.dup();
        clone.gammaMatrix = gammaMatrix.dup();
        clone.betaMatrix = betaMatrix.dup();
        return clone;
    }

    @Override
    public List<DoubleMatrix> getParametersMatrix() {
        return Arrays.asList(getWeightMatrix(), getGammaMatrix(), getBetaMatrix());
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        BatchNormLayer that = (BatchNormLayer) o;
        return inputLayerSize == that.inputLayerSize &&
                outputLayerSize == that.outputLayerSize &&
                Objects.equals(weightMatrix, that.weightMatrix) &&
                Objects.equals(gammaMatrix, that.gammaMatrix) &&
                Objects.equals(betaMatrix, that.betaMatrix);
    }

    @Override
    public int hashCode() {
        return Objects.hash(inputLayerSize, outputLayerSize, weightMatrix, gammaMatrix, betaMatrix);
    }

    @Override
    public String toString() {
        return "BatchNormLayer{" +
                "inputLayerSize=" + inputLayerSize +
                ", outputLayerSize=" + outputLayerSize +
                ", weightMatrix=" + weightMatrix +
                ", gammaMatrix=" + gammaMatrix +
                ", betaMatrix=" + betaMatrix +
                '}';
    }

    @Override
    public void accept(LayerVisitor layerVisitor) {
        layerVisitor.visit(this);
    }
}
