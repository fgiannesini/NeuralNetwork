package com.fgiannesini.neuralnetwork.activationfunctions;

import org.jblas.FloatMatrix;
import org.jblas.MatrixFunctions;

public class SigmoidFunctionApplyer implements ActivationFunctionApplyer {

    @Override
    public FloatMatrix apply(FloatMatrix input) {
        //1/(1+exp(-z))
        FloatMatrix output = input.mul(-1);
        output = MatrixFunctions.expi(output);
        output = output.addi(1);
        return MatrixFunctions.powi(output, -1);
    }
}
