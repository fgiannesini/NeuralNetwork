package com.fgiannesini.neuralnetwork.activationfunctions;

import org.jblas.FloatMatrix;

public class LeakyReluFunctionApplier implements ActivationFunctionApplier {

    @Override
    public FloatMatrix apply(FloatMatrix input) {
        return input.max(input.mul(0.01f));
    }
}
