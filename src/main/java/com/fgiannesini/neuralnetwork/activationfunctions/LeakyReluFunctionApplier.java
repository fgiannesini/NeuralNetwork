package com.fgiannesini.neuralnetwork.activationfunctions;

import org.jblas.DoubleMatrix;

public class LeakyReluFunctionApplier implements ActivationFunctionApplier {

    @Override
    public DoubleMatrix apply(DoubleMatrix input) {
        return input.max(input.mul(0.01f));
    }

    @Override
    public DoubleMatrix derivate(DoubleMatrix input, DoubleMatrix previousError) {
        // if input>=0 1, else -0.01
        DoubleMatrix greaterEqualZero = input.gt(0);
        DoubleMatrix lesserThanZero = input.le(0).negi().muli(0.01f);
        return greaterEqualZero.addi(lesserThanZero).mul(previousError);
    }
}
