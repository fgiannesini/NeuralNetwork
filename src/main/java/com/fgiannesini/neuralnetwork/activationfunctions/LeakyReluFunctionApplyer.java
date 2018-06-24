package com.fgiannesini.neuralnetwork.activationfunctions;

import org.jblas.FloatMatrix;

public class LeakyReluFunctionApplyer implements ActivationFunctionApplyer {

    @Override
    public FloatMatrix apply(FloatMatrix input) {
        return input.max(input.mul(0.1f));
    }
}
