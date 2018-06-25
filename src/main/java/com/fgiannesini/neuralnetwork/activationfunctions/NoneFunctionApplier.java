package com.fgiannesini.neuralnetwork.activationfunctions;

import org.jblas.FloatMatrix;

public class NoneFunctionApplier implements ActivationFunctionApplier {

    @Override
    public FloatMatrix apply(FloatMatrix input) {
        return input;
    }

    @Override
    public FloatMatrix derivate(FloatMatrix input) {
       return input;
    }
}
