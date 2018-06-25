package com.fgiannesini.neuralnetwork.activationfunctions;

import org.jblas.FloatMatrix;

public class ReluFunctionApplier implements ActivationFunctionApplier {

    @Override
    public FloatMatrix apply(FloatMatrix input) {
        return input.max(0);
    }

    @Override
    public FloatMatrix derivate(FloatMatrix input) {
        // if input>=0 1, else 0
        return input.ge(0);
    }
}
