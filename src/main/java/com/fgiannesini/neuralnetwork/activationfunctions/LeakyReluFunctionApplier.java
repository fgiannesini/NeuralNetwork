package com.fgiannesini.neuralnetwork.activationfunctions;

import org.jblas.FloatMatrix;

public class LeakyReluFunctionApplier implements ActivationFunctionApplier {

    @Override
    public FloatMatrix apply(FloatMatrix input) {
        return input.max(input.mul(0.01f));
    }

    @Override
    public FloatMatrix derivate(FloatMatrix input) {
        // if input>=0 1, else -0.01
        FloatMatrix greaterEqualZero = input.ge(0);
        FloatMatrix lesserThanZero = input.lt(0).negi().muli(0.01f);
        return greaterEqualZero.addi(lesserThanZero);
    }
}
