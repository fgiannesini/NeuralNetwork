package com.fgiannesini.neuralnetwork.initializer;

import org.jblas.DoubleMatrix;

public class RandomInitializer implements Initializer {

    @Override
    public DoubleMatrix initDoubleMatrix(int inputSize, int outputSize) {
        return DoubleMatrix.rand(inputSize, outputSize).muli(0.01);
    }
}
