package com.fgiannesini.neuralnetwork.cost;

import org.jblas.DoubleMatrix;

public interface CostComputer {

    default double compute(double[] input, double[] output) {
        return compute(new DoubleMatrix(input), new DoubleMatrix(output));
    }

    default double compute(double[][] input, double[][] output) {
        return compute(new DoubleMatrix(input).transpose(), new DoubleMatrix(output).transpose());
    }

    double compute(DoubleMatrix input, DoubleMatrix output);
}
