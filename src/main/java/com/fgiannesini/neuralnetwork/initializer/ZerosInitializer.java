package com.fgiannesini.neuralnetwork.initializer;

import org.jblas.DoubleMatrix;

public class ZerosInitializer implements Initializer {

    @Override
    public DoubleMatrix initDoubleMatrix(int inputSize, int outputSize) {
        return DoubleMatrix.zeros(inputSize, outputSize);
    }
}
