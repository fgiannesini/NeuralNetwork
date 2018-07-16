package com.fgiannesini.neuralnetwork.initializer;

import org.jblas.DoubleMatrix;

public class UniformInitializer implements Initializer {

    @Override
    public DoubleMatrix initDoubleMatrix(int inputSize, int outputSize) {
        return DoubleMatrix.rand(inputSize, outputSize).muli(Math.sqrt(1 / (double) inputSize));
    }
}
