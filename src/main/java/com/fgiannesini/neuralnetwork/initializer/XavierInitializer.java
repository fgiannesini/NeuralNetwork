package com.fgiannesini.neuralnetwork.initializer;

import org.jblas.DoubleMatrix;

public class XavierInitializer implements Initializer {
    @Override
    public DoubleMatrix initDoubleMatrix(int inputSize, int outputSize) {
        return DoubleMatrix.rand(inputSize, outputSize).muli(Math.sqrt(2 / (double) (inputSize + outputSize)));
    }
}
