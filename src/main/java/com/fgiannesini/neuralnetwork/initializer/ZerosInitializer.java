package com.fgiannesini.neuralnetwork.initializer;

import org.jblas.FloatMatrix;

public class ZerosInitializer implements Initializer {

    @Override
    public FloatMatrix initFloatMatrix(int inputSize, int outputSize) {
        return FloatMatrix.zeros(inputSize, outputSize);
    }
}
