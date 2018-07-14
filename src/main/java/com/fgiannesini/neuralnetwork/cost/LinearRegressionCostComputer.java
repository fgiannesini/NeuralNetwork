package com.fgiannesini.neuralnetwork.cost;

import com.fgiannesini.neuralnetwork.computer.finaloutputcomputer.IFinalOutputComputer;
import org.jblas.DoubleMatrix;

public class LinearRegressionCostComputer implements CostComputer {

    private final IFinalOutputComputer outputComputer;

    public LinearRegressionCostComputer(IFinalOutputComputer outputComputer) {
        this.outputComputer = outputComputer;
    }

    @Override
    public double compute(DoubleMatrix input, DoubleMatrix output) {
        DoubleMatrix computedOutput = outputComputer.compute(input);
        double inputCount = input.getColumns();
        return computedOutput.squaredDistance(output) / (inputCount * 2d);
    }
}
