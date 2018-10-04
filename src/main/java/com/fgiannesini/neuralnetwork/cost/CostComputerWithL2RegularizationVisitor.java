package com.fgiannesini.neuralnetwork.cost;

import com.fgiannesini.neuralnetwork.computer.BatchNormData;
import com.fgiannesini.neuralnetwork.computer.DataVisitor;
import com.fgiannesini.neuralnetwork.computer.WeightBiasData;
import com.fgiannesini.neuralnetwork.model.BatchNormLayer;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import com.fgiannesini.neuralnetwork.model.WeightBiasLayer;
import org.jblas.MatrixFunctions;

public class CostComputerWithL2RegularizationVisitor implements DataVisitor {

    private final NeuralNetworkModel neuralNetworkModel;
    private final double regularizationCoeff;
    private double cost;

    public CostComputerWithL2RegularizationVisitor(NeuralNetworkModel neuralNetworkModel, double regularizationCoeff) {
        this.neuralNetworkModel = neuralNetworkModel;
        this.regularizationCoeff = regularizationCoeff;
    }

    @Override
    public void visit(WeightBiasData data) {
        //sum(Wij²) * lambda / 2m
        double squaredWeightsSum = neuralNetworkModel.getLayers().stream()
                .mapToDouble(layer -> MatrixFunctions.pow(((WeightBiasLayer) layer).getWeightMatrix(), 2).sum())
                .sum();
        double inputCount = data.getData().getColumns();
        cost = squaredWeightsSum * regularizationCoeff / 2d / inputCount;
    }

    @Override
    public void visit(BatchNormData data) {
//sum(Wij²) * lambda / 2m
        double squaredWeightsSum = neuralNetworkModel.getLayers().stream()
                .mapToDouble(layer -> MatrixFunctions.pow(((BatchNormLayer) layer).getWeightMatrix(), 2).sum())
                .sum();
        double inputCount = data.getData().getColumns();
        cost = squaredWeightsSum * regularizationCoeff / 2d / inputCount;
    }

    public double getCost() {
        return cost;
    }
}
