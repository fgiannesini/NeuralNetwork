package com.fgiannesini.neuralnetwork.learningalgorithm.regularization.dropout;

import com.fgiannesini.neuralnetwork.model.*;
import org.jblas.DoubleMatrix;

public class FirstDropOutMatrixGeneratorVisitor implements LayerVisitor {

    private final double firstDropOutParameter;
    private DoubleMatrix firstDropOutMatrix;

    public FirstDropOutMatrixGeneratorVisitor(double firstDropOutParameter) {
        this.firstDropOutParameter = firstDropOutParameter;
    }

    private void buildFirstDropOutMatrix(int inputLayerSize) {
        firstDropOutMatrix = DoubleMatrix.rand(1, inputLayerSize).lei(firstDropOutParameter).divi(firstDropOutParameter);
    }

    @Override
    public void visit(WeightBiasLayer layer) {
        buildFirstDropOutMatrix(layer.getInputLayerSize());
    }

    @Override
    public void visit(BatchNormLayer layer) {
        buildFirstDropOutMatrix(layer.getInputLayerSize());
    }

    @Override
    public void visit(AveragePoolingLayer layer) {

    }

    @Override
    public void visit(MaxPoolingLayer layer) {

    }

    @Override
    public void visit(ConvolutionLayer layer) {

    }

    public DoubleMatrix getFirstDropOutMatrix() {
        return firstDropOutMatrix;
    }
}
