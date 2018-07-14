package com.fgiannesini.neuralnetwork.activationfunctions;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

public class IdentityFunctionApplier implements ActivationFunctionApplier {

    @Override
    public DoubleMatrix apply(DoubleMatrix input) {
        return input.dup();
    }

    @Override
    public DoubleMatrix derivate(DoubleMatrix input) {
        return MatrixFunctions.signum(input);
    }
}
