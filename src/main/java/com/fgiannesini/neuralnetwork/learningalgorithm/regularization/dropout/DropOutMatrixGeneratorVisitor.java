package com.fgiannesini.neuralnetwork.learningalgorithm.regularization.dropout;

import com.fgiannesini.neuralnetwork.model.*;
import org.jblas.DoubleMatrix;

public class DropOutMatrixGeneratorVisitor implements LayerVisitor {

    private final double dropOutParameter;
    private DoubleMatrix dropOutMatrix;

    public DropOutMatrixGeneratorVisitor(double dropOutParameter) {
        this.dropOutParameter = dropOutParameter;
    }

    public void buildDropOutMatrix(int outputLayerSize) {
        dropOutMatrix = DoubleMatrix.rand(1, outputLayerSize).lei(dropOutParameter).divi(dropOutParameter);
    }

    @Override
    public void visit(WeightBiasLayer layer) {
        buildDropOutMatrix(layer.getOutputLayerSize());
    }

    @Override
    public void visit(BatchNormLayer layer) {
        buildDropOutMatrix(layer.getOutputLayerSize());
    }

    @Override
    public void visit(AveragePoolingLayer layer) {
        dropOutMatrix = DoubleMatrix.EMPTY;
    }

    @Override
    public void visit(MaxPoolingLayer layer) {
        dropOutMatrix = DoubleMatrix.EMPTY;
    }

    @Override
    public void visit(ConvolutionLayer layer) {
        dropOutMatrix = DoubleMatrix.rand(layer.getOutputHeight(), layer.getOutputWidth()).lei(dropOutParameter).divi(dropOutParameter);
    }

    public DoubleMatrix getDropOutMatrix() {
        return dropOutMatrix;
    }
}
