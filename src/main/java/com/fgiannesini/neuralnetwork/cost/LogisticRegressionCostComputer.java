package com.fgiannesini.neuralnetwork.cost;

import com.fgiannesini.neuralnetwork.computer.data.LayerTypeData;
import com.fgiannesini.neuralnetwork.computer.finaloutputcomputer.IFinalOutputComputer;

public class LogisticRegressionCostComputer implements CostComputer {

    private final IFinalOutputComputer outputComputer;

    public LogisticRegressionCostComputer(IFinalOutputComputer outputComputer) {
        this.outputComputer = outputComputer;
    }

    @Override
    public double compute(LayerTypeData input, LayerTypeData output) {
        LogisticRegressionCostComputerVisitor computerVisitor = new LogisticRegressionCostComputerVisitor(input, outputComputer);
        output.accept(computerVisitor);
        return computerVisitor.getCost();
    }
}
