package com.fgiannesini.neuralnetwork.cost;

import com.fgiannesini.neuralnetwork.computer.data.*;
import com.fgiannesini.neuralnetwork.computer.finaloutputcomputer.IFinalOutputComputer;

import java.util.stream.IntStream;

public class LinearRegressionCostComputerVisitor implements DataVisitor {

    private final LayerTypeData input;
    private final IFinalOutputComputer outputComputer;
    private double cost;

    public LinearRegressionCostComputerVisitor(LayerTypeData input, IFinalOutputComputer outputComputer) {
        this.input = input;
        this.outputComputer = outputComputer;
    }

    @Override
    public void visit(WeightBiasData output) {
        WeightBiasData computedOutput = (WeightBiasData) outputComputer.compute(this.input);
        double inputCount = computedOutput.getData().getColumns();
        cost = computedOutput.getData().squaredDistance(output.getData()) / (inputCount * 2d);
    }

    @Override
    public void visit(BatchNormData output) {
        BatchNormData computedOutput = (BatchNormData) outputComputer.compute(input);
        double inputCount = computedOutput.getData().getColumns();
        cost = computedOutput.getData().squaredDistance(output.getData()) / (inputCount * 2d);
    }

    @Override
    public void visit(ConvolutionData output) {
        ConvolutionData computedOutput = (ConvolutionData) outputComputer.compute(input);
        int inputCount = computedOutput.getDatas().size();
        cost = IntStream.range(0, inputCount)
                .mapToDouble(i -> computedOutput.getDatas().get(i).squaredDistance(output.getDatas().get(i)))
                .sum() / (inputCount * 2d);
    }

    public double getCost() {
        return cost;
    }
}
