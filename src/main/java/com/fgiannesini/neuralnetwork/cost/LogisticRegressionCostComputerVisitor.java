package com.fgiannesini.neuralnetwork.cost;

import com.fgiannesini.neuralnetwork.computer.BatchNormData;
import com.fgiannesini.neuralnetwork.computer.DataVisitor;
import com.fgiannesini.neuralnetwork.computer.LayerTypeData;
import com.fgiannesini.neuralnetwork.computer.WeightBiasData;
import com.fgiannesini.neuralnetwork.computer.finaloutputcomputer.IFinalOutputComputer;

public class LogisticRegressionCostComputerVisitor implements DataVisitor {


    private final LayerTypeData output;
    private final IFinalOutputComputer outputComputer;
    private double cost;

    public LogisticRegressionCostComputerVisitor(LayerTypeData output, IFinalOutputComputer outputComputer) {
        this.output = output;
        this.outputComputer = outputComputer;
    }

    @Override
    public void visit(WeightBiasData data) {
        double inputCount = data.getData().columns;
        WeightBiasData computedOutput = (WeightBiasData) outputComputer.compute(data);
        WeightBiasData output = (WeightBiasData) this.output;
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
    public void visit(BatchNormData data) {
        double inputCount = data.getData().columns;
        BatchNormData computedOutput = (BatchNormData) outputComputer.compute(data);
        BatchNormData output = (BatchNormData) this.output;
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
