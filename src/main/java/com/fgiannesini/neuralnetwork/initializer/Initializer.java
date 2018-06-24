package com.fgiannesini.neuralnetwork.initializer;

import org.jblas.FloatMatrix;

public interface Initializer {

    FloatMatrix initFloatMatrix(int inputSize, int outputSize);
}
