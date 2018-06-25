package com.fgiannesini.neuralnetwork.activationfunctions;

import org.jblas.FloatMatrix;
import org.jblas.MatrixFunctions;

public class TanhFunctionApplier implements ActivationFunctionApplier {

    @Override
    public FloatMatrix apply(FloatMatrix input) {
        return MatrixFunctions.tanh(input);
    }

    @Override
    public FloatMatrix derivate(FloatMatrix input) {
        //1-a²
        return MatrixFunctions.pow(input, 2).negi().addi(1);
    }
}
