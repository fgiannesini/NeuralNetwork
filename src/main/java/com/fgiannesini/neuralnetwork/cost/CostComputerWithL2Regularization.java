package com.fgiannesini.neuralnetwork.cost;

import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import com.fgiannesini.neuralnetwork.model.WeightBiasLayer;
import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

public class CostComputerWithL2Regularization implements CostComputer {

    private final NeuralNetworkModel<WeightBiasLayer> neuralNetworkModel;
    private final CostComputer costComputer;
    private final double regularizationCoeff;

    public CostComputerWithL2Regularization(NeuralNetworkModel neuralNetworkModel, CostComputer costComputer, double regularizationCoeff) {
        this.neuralNetworkModel = neuralNetworkModel;
        this.costComputer = costComputer;
        this.regularizationCoeff = regularizationCoeff;
    }

    @Override
    public double compute(DoubleMatrix input, DoubleMatrix output) {
        double costWithoutLinearRegression = costComputer.compute(input, output);

        //sum(Wij²) * lambda / 2m
        double squaredWeightsSum = neuralNetworkModel.getLayers().stream()
                .mapToDouble(layer -> MatrixFunctions.pow(layer.getWeightMatrix(), 2).sum())
                .sum();
        double inputCount = input.getColumns();
        double regularizationCost = squaredWeightsSum * regularizationCoeff / 2d / inputCount;
        return costWithoutLinearRegression + regularizationCost;
    }
}
