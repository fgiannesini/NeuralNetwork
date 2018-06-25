package com.fgiannesini.neuralnetwork.activationfunctions;

import org.jblas.FloatMatrix;

public class IdentityFunctionApplier implements ActivationFunctionApplier {

    @Override
    public FloatMatrix apply(FloatMatrix input) {
        return input.dup();
    }

    @Override
    public FloatMatrix derivate(FloatMatrix input) {
        return input.dup();
    }
}
