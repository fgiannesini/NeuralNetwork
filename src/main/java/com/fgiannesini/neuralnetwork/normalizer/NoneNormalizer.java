package com.fgiannesini.neuralnetwork.normalizer;

import org.jblas.DoubleMatrix;

public class NoneNormalizer implements INormalizer {

    @Override
    public DoubleMatrix normalize(DoubleMatrix input) {
        return input.dup();
    }
}
