package com.fgiannesini.neuralnetwork.cost;

import com.fgiannesini.neuralnetwork.model.*;
import org.jblas.MatrixFunctions;

public class CostComputerWithL2RegularizationVisitor implements LayerVisitor {

    private int inputCount;
    private final double regularizationCoeff;
    private double cost;

    public CostComputerWithL2RegularizationVisitor(int inputCount, double regularizationCoeff) {
        this.inputCount = inputCount;
        this.regularizationCoeff = regularizationCoeff;
    }

    @Override
    public void visit(WeightBiasLayer layer) {
        //sum(Wij²) * lambda / 2m
        double squaredWeightsSum = MatrixFunctions.pow(layer.getWeightMatrix(), 2).sum();
        cost = squaredWeightsSum * regularizationCoeff / 2d / inputCount;
    }

    @Override
    public void visit(BatchNormLayer layer) {
        //sum(Wij²) * lambda / 2m
        double squaredWeightsSum = MatrixFunctions.pow(layer.getWeightMatrix(), 2).sum();
        cost = squaredWeightsSum * regularizationCoeff / 2d / inputCount;
    }

    @Override
    public void visit(ConvolutionLayer layer) {
        //sum(Wij²) * lambda / 2m
        double squaredWeightsSum = layer.getWeightMatrices().stream()
                .mapToDouble(weights -> MatrixFunctions.pow(weights, 2).sum())
                .sum();
        cost = squaredWeightsSum * regularizationCoeff / 2d / inputCount;
    }

    @Override
    public void visit(AveragePoolingLayer layer) {

    }

    @Override
    public void visit(MaxPoolingLayer layer) {

    }

    public double getCost() {
        return cost;
    }
}
