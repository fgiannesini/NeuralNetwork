package com.fgiannesini.neuralnetwork.initializer;

import org.jblas.DoubleMatrix;

public class OnesInitializer implements Initializer {
    @Override
    public DoubleMatrix initDoubleMatrix(int inputSize, int outputSize) {
        return DoubleMatrix.ones(inputSize, outputSize);
    }
}
