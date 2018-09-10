package com.fgiannesini.neuralnetwork.activationfunctions;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

public class TanhFunctionApplier implements ActivationFunctionApplier {

    @Override
    public DoubleMatrix apply(DoubleMatrix input) {
        return MatrixFunctions.tanh(input);
    }

    @Override
    public DoubleMatrix derivate(DoubleMatrix input, DoubleMatrix previousError) {
        //1-a²
        return MatrixFunctions.pow(input, 2).negi().addi(1).muli(previousError);
    }
}
