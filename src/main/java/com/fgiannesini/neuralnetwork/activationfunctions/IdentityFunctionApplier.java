package com.fgiannesini.neuralnetwork.activationfunctions;

import org.jblas.DoubleMatrix;

public class IdentityFunctionApplier implements ActivationFunctionApplier {

    @Override
    public DoubleMatrix apply(DoubleMatrix input) {
        return input.dup();
    }

    @Override
    public DoubleMatrix derivate(DoubleMatrix input) {
        return input.dup();
    }
}
