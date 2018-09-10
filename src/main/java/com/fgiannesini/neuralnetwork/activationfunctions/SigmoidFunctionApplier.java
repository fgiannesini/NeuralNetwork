package com.fgiannesini.neuralnetwork.activationfunctions;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

public class SigmoidFunctionApplier implements ActivationFunctionApplier {

    @Override
    public DoubleMatrix apply(DoubleMatrix input) {
        //1/(1+exp(-z))
        DoubleMatrix output = input.mul(-1);
        MatrixFunctions.expi(output);
        output.addi(1);
        return MatrixFunctions.powi(output, -1);
    }

    @Override
    public DoubleMatrix derivate(DoubleMatrix input, DoubleMatrix previousError) {
        //(1-a)*a
        return input.neg().addi(1).muli(input).mul(previousError);
    }
}
