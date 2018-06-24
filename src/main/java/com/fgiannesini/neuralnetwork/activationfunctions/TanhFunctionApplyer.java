package com.fgiannesini.neuralnetwork.activationfunctions;

import org.jblas.FloatMatrix;
import org.jblas.MatrixFunctions;

public class TanhFunctionApplyer implements ActivationFunctionApplyer {

    @Override
    public FloatMatrix apply(FloatMatrix input) {
        return MatrixFunctions.tanh(input);
    }
}
