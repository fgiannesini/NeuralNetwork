package com.fgiannesini.neuralnetwork.cost;

import com.fgiannesini.neuralnetwork.computer.LayerTypeData;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;

public class CostComputerWithL2Regularization implements CostComputer {

    private final NeuralNetworkModel neuralNetworkModel;
    private final CostComputer costComputer;
    private final double regularizationCoeff;

    public CostComputerWithL2Regularization(NeuralNetworkModel neuralNetworkModel, CostComputer costComputer, double regularizationCoeff) {
        this.neuralNetworkModel = neuralNetworkModel;
        this.costComputer = costComputer;
        this.regularizationCoeff = regularizationCoeff;
    }

    @Override
    public double compute(LayerTypeData input, LayerTypeData output) {
        double costWithoutLinearRegression = costComputer.compute(input, output);
        CostComputerWithL2RegularizationVisitor regularizationVisitor = new CostComputerWithL2RegularizationVisitor(neuralNetworkModel, regularizationCoeff);
        input.accept(regularizationVisitor);
        return costWithoutLinearRegression + regularizationVisitor.getCost();
    }

}
