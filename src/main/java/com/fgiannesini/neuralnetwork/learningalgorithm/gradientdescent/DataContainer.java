package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent;

import org.jblas.DoubleMatrix;

public class DataContainer {

    private final DoubleMatrix input;
    private final DoubleMatrix output;

    public DataContainer(DoubleMatrix input, DoubleMatrix output) {
        this.input = input;
        this.output = output;
    }

    public DoubleMatrix getInput() {
        return input;
    }

    public DoubleMatrix getOutput() {
        return output;
    }
}
