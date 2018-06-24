package com.fgiannesini.neuralnetwork.initializer;

import org.jblas.FloatMatrix;

public class RandomInitializer implements Initializer {

    @Override
    public FloatMatrix initFloatMatrix(int inputSize, int outputSize) {
        return FloatMatrix.rand(inputSize, outputSize);
    }
}
