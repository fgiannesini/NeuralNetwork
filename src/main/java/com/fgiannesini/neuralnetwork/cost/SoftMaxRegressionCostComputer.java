package com.fgiannesini.neuralnetwork.cost;

import com.fgiannesini.neuralnetwork.computer.finaloutputcomputer.IFinalOutputComputer;
import org.jblas.DoubleMatrix;

public class SoftMaxRegressionCostComputer implements CostComputer {

    private final IFinalOutputComputer outputComputer;
    private final double epsilon;

    public SoftMaxRegressionCostComputer(IFinalOutputComputer outputComputer) {
        this.outputComputer = outputComputer;
        this.epsilon = Math.pow(10, -18);
    }

    @Override
    public double compute(DoubleMatrix input, DoubleMatrix output) {
        double inputCount = input.columns;
        DoubleMatrix computedOutput = outputComputer.compute(input);
        //cost = -1/m sum(y * log(^y))
        double result = 0;
        for (int index = 0; index < output.length; index++) {
            double outputValue = output.get(index);
            double computedOutputValue = computedOutput.get(index);
            if (outputValue != 0) {
                result += outputValue * Math.log(computedOutputValue + epsilon);
            }
        }
        return -result / inputCount;
    }
}
