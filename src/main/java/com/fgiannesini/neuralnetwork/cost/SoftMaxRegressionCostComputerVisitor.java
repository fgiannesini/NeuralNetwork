package com.fgiannesini.neuralnetwork.cost;

import com.fgiannesini.neuralnetwork.computer.data.BatchNormData;
import com.fgiannesini.neuralnetwork.computer.data.DataVisitor;
import com.fgiannesini.neuralnetwork.computer.data.LayerTypeData;
import com.fgiannesini.neuralnetwork.computer.data.WeightBiasData;
import com.fgiannesini.neuralnetwork.computer.finaloutputcomputer.IFinalOutputComputer;

public class SoftMaxRegressionCostComputerVisitor implements DataVisitor {


    private final LayerTypeData output;
    private final IFinalOutputComputer outputComputer;
    private final double epsilon;
    private double cost;

    public SoftMaxRegressionCostComputerVisitor(LayerTypeData output, IFinalOutputComputer outputComputer) {
        this.output = output;
        this.outputComputer = outputComputer;
        epsilon = Math.pow(10, -18);
    }

    @Override
    public void visit(WeightBiasData data) {
        double inputCount = data.getData().columns;
        WeightBiasData computedOutput = (WeightBiasData) outputComputer.compute(data);
        WeightBiasData output = (WeightBiasData) this.output;
        //cost = -1/m sum(y * log(^y))
        double result = 0;
        for (int index = 0; index < output.getData().length; index++) {
            double outputValue = output.getData().get(index);
            double computedOutputValue = computedOutput.getData().get(index);
            if (outputValue != 0) {
                result += outputValue * Math.log(computedOutputValue + epsilon);
            }
        }
        cost = -result / inputCount;
    }

    @Override
    public void visit(BatchNormData data) {
        double inputCount = data.getData().columns;
        BatchNormData computedOutput = (BatchNormData) outputComputer.compute(data);
        BatchNormData output = (BatchNormData) this.output;
        //cost = -1/m sum(y * log(^y))
        double result = 0;
        for (int index = 0; index < output.getData().length; index++) {
            double outputValue = output.getData().get(index);
            double computedOutputValue = computedOutput.getData().get(index);
            if (outputValue != 0) {
                result += outputValue * Math.log(computedOutputValue + epsilon);
            }
        }
        cost = -result / inputCount;
    }

    public double getCost() {
        return cost;
    }
}
