package com.fgiannesini.neuralnetwork.activationfunctions;

import org.jblas.FloatMatrix;

public class ReluFunctionApplyer implements ActivationFunctionApplyer {

    @Override
    public FloatMatrix apply(FloatMatrix input) {
        return input.max(0);
    }
}
