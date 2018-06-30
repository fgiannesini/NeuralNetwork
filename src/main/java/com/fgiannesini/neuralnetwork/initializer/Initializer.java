package com.fgiannesini.neuralnetwork.initializer;

import org.jblas.DoubleMatrix;

public interface Initializer {

    DoubleMatrix initDoubleMatrix(int inputSize, int outputSize);
}
