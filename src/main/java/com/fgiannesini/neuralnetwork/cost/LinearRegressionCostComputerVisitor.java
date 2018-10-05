package com.fgiannesini.neuralnetwork.cost;

import com.fgiannesini.neuralnetwork.computer.data.BatchNormData;
import com.fgiannesini.neuralnetwork.computer.data.DataVisitor;
import com.fgiannesini.neuralnetwork.computer.data.LayerTypeData;
import com.fgiannesini.neuralnetwork.computer.data.WeightBiasData;
import com.fgiannesini.neuralnetwork.computer.finaloutputcomputer.IFinalOutputComputer;

public class LinearRegressionCostComputerVisitor implements DataVisitor {


    private final LayerTypeData output;
    private final IFinalOutputComputer outputComputer;
    private double cost;

    public LinearRegressionCostComputerVisitor(LayerTypeData output, IFinalOutputComputer outputComputer) {
        this.output = output;
        this.outputComputer = outputComputer;
    }

    @Override
    public void visit(WeightBiasData data) {
        WeightBiasData computedOutput = (WeightBiasData) outputComputer.compute(data);
        WeightBiasData output = (WeightBiasData) this.output;
        double inputCount = computedOutput.getData().getColumns();
        cost = computedOutput.getData().squaredDistance(output.getData()) / (inputCount * 2d);
    }

    @Override
    public void visit(BatchNormData data) {
        BatchNormData computedOutput = (BatchNormData) outputComputer.compute(data);
        BatchNormData output = (BatchNormData) this.output;
        double inputCount = computedOutput.getData().getColumns();
        cost = computedOutput.getData().squaredDistance(output.getData()) / (inputCount * 2d);
    }

    public double getCost() {
        return cost;
    }
}
