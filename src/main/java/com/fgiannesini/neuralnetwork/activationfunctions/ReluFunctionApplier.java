package com.fgiannesini.neuralnetwork.activationfunctions;

import org.jblas.DoubleMatrix;

public class ReluFunctionApplier implements ActivationFunctionApplier {

    @Override
    public DoubleMatrix apply(DoubleMatrix input) {
        return input.max(0);
    }

    @Override
    public DoubleMatrix derivate(DoubleMatrix input) {
        // if input>=0 1, else 0
        return input.ge(0);
    }
}
