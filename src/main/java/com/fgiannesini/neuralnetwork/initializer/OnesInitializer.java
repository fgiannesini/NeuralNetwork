package com.fgiannesini.neuralnetwork.initializer;

import org.jblas.FloatMatrix;

public class OnesInitializer implements Initializer {
    @Override
    public FloatMatrix initFloatMatrix(int inputSize, int outputSize) {
        return FloatMatrix.ones(inputSize, outputSize);
    }
}
