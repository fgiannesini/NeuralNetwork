package com.fgiannesini.neuralnetwork.cost;

import com.fgiannesini.neuralnetwork.computer.LayerTypeData;
import com.fgiannesini.neuralnetwork.computer.finaloutputcomputer.IFinalOutputComputer;

public class LinearRegressionCostComputer implements CostComputer {

    private final IFinalOutputComputer outputComputer;

    public LinearRegressionCostComputer(IFinalOutputComputer outputComputer) {
        this.outputComputer = outputComputer;
    }

    @Override
    public double compute(LayerTypeData input, LayerTypeData output) {
        LinearRegressionCostComputerVisitor computerVisitor = new LinearRegressionCostComputerVisitor(output, outputComputer);
        input.accept(computerVisitor);
        return computerVisitor.getCost();
    }
}
