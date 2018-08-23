package com.fgiannesini.neuralnetwork.model;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import com.fgiannesini.neuralnetwork.initializer.Initializer;
import org.jblas.DoubleMatrix;

import java.util.Objects;

public class BatchNormLayer extends Layer implements Cloneable {

    private DoubleMatrix gammaMatrix;
    private DoubleMatrix betaMatrix;

    public BatchNormLayer(int inputLayerSize, int outputLayerSize, Initializer initializer, ActivationFunctionType activationFunctionType) {
        super(inputLayerSize, outputLayerSize, activationFunctionType, initializer);
        gammaMatrix = initializer.initDoubleMatrix(outputLayerSize, 1);
        betaMatrix = initializer.initDoubleMatrix(outputLayerSize, 1);
    }

    public DoubleMatrix getGammaMatrix() {
        return gammaMatrix;
    }

    public DoubleMatrix getBetaMatrix() {
        return betaMatrix;
    }

    @Override
    public BatchNormLayer clone() {
        BatchNormLayer clone = (BatchNormLayer) super.clone();
        clone.gammaMatrix = gammaMatrix.dup();
        clone.betaMatrix = betaMatrix.dup();
        return clone;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof BatchNormLayer)) return false;
        if (!super.equals(o)) return false;
        BatchNormLayer that = (BatchNormLayer) o;
        return Objects.equals(gammaMatrix, that.gammaMatrix) &&
                Objects.equals(betaMatrix, that.betaMatrix);
    }

    @Override
    public int hashCode() {
        return Objects.hash(super.hashCode(), gammaMatrix, betaMatrix);
    }

    @Override
    public String toString() {
        return "BatchNormLayer{" +
                "gammaMatrix=" + gammaMatrix +
                ", betaMatrix=" + betaMatrix +
                "} " + super.toString();
    }
}
