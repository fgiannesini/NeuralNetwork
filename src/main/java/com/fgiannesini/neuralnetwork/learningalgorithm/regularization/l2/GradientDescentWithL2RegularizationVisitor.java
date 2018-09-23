package com.fgiannesini.neuralnetwork.learningalgorithm.regularization.l2;

import com.fgiannesini.neuralnetwork.model.*;
import org.jblas.DoubleMatrix;

import java.util.List;
import java.util.stream.IntStream;

public class GradientDescentWithL2RegularizationVisitor implements LayerVisitor {

    private final Layer originalLayer;
    private final double regularizationCoeff;
    private final double learningRate;
    private final int inputCount;

    public GradientDescentWithL2RegularizationVisitor(Layer originalLayer, double regularizationCoeff, double learningRate, int inputCount) {
        this.originalLayer = originalLayer;
        this.regularizationCoeff = regularizationCoeff;
        this.learningRate = learningRate;
        this.inputCount = inputCount;
    }

    @Override
    public void visit(WeightBiasLayer layer) {
        DoubleMatrix originalWeightMatrix = ((WeightBiasLayer) originalLayer).getWeightMatrix();
        layer.getWeightMatrix().subi(originalWeightMatrix.mul(learningRate * regularizationCoeff / inputCount));
    }

    @Override
    public void visit(BatchNormLayer layer) {
        DoubleMatrix originalWeightMatrix = ((BatchNormLayer) originalLayer).getWeightMatrix();
        layer.getWeightMatrix().subi(originalWeightMatrix.mul(learningRate * regularizationCoeff / inputCount));
    }

    @Override
    public void visit(AveragePoolingLayer layer) {
    }

    @Override
    public void visit(MaxPoolingLayer layer) {
    }

    @Override
    public void visit(ConvolutionLayer layer) {
        List<DoubleMatrix> originalWeightMatrix = ((ConvolutionLayer) originalLayer).getWeightMatrices();
        IntStream.range(0, originalWeightMatrix.size()).forEach(i -> layer.getWeightMatrices().get(i).subi(originalWeightMatrix.get(i).mul(learningRate * regularizationCoeff / inputCount)));
    }
}
