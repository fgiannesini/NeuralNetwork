package com.fgiannesini.neuralnetwork.cost;

import com.fgiannesini.neuralnetwork.computer.data.BatchNormData;
import com.fgiannesini.neuralnetwork.computer.data.DataVisitor;
import com.fgiannesini.neuralnetwork.computer.data.LayerTypeData;
import com.fgiannesini.neuralnetwork.computer.data.WeightBiasData;
import com.fgiannesini.neuralnetwork.computer.finaloutputcomputer.IFinalOutputComputer;

public class LogisticRegressionCostComputerVisitor implements DataVisitor {

    private final LayerTypeData input;
    private final IFinalOutputComputer outputComputer;
    private double cost;

    public LogisticRegressionCostComputerVisitor(LayerTypeData input, IFinalOutputComputer outputComputer) {
        this.input = input;
        this.outputComputer = outputComputer;
    }

    @Override
    public void visit(WeightBiasData output) {
        double inputCount = output.getData().columns;
        WeightBiasData computedOutput = (WeightBiasData) outputComputer.compute(input);
        //cost = -1/m sum(y * log(^y) + (1-y) * log (1-^y))
        double result = 0;
        for (int index = 0; index < output.getData().length; index++) {
            double outputValue = output.getData().get(index);
            double computedOutputValue = computedOutput.getData().get(index);
            if (outputValue != 0) {
                result += outputValue * Math.log(computedOutputValue);
            }
            if (outputValue != 1) {
                result += (1 - outputValue) * Math.log(1 - computedOutputValue);
            }
        }
        cost = -result / inputCount;
    }

    @Override
    public void visit(BatchNormData output) {
        double inputCount = output.getData().columns;
        BatchNormData computedOutput = (BatchNormData) outputComputer.compute(input);

        //cost = -1/m sum(y * log(^y) + (1-y) * log (1-^y))
        double result = 0;
        for (int index = 0; index < output.getData().length; index++) {
            double outputValue = output.getData().get(index);
            double computedOutputValue = computedOutput.getData().get(index);
            if (outputValue != 0) {
                result += outputValue * Math.log(computedOutputValue);
            }
            if (outputValue != 1) {
                result += (1 - outputValue) * Math.log(1 - computedOutputValue);
            }
        }
        cost = -result / inputCount;
    }

    public double getCost() {
        return cost;
    }
}
