package com.fgiannesini.neuralnetwork.cost;

import com.fgiannesini.neuralnetwork.computer.LayerTypeData;
import com.fgiannesini.neuralnetwork.computer.finaloutputcomputer.IFinalOutputComputer;

public class SoftMaxRegressionCostComputer implements CostComputer {

    private final IFinalOutputComputer outputComputer;

    public SoftMaxRegressionCostComputer(IFinalOutputComputer outputComputer) {
        this.outputComputer = outputComputer;
    }

    @Override
    public double compute(LayerTypeData input, LayerTypeData output) {
        SoftMaxRegressionCostComputerVisitor computerVisitor = new SoftMaxRegressionCostComputerVisitor(output, outputComputer);
        input.accept(computerVisitor);
        return computerVisitor.getCost();
    }
}
