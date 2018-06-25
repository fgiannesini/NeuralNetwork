package com.fgiannesini.neuralnetwork.activationfunctions;

import org.jblas.FloatMatrix;
import org.jblas.MatrixFunctions;

public class SigmoidFunctionApplier implements ActivationFunctionApplier {

    @Override
    public FloatMatrix apply(FloatMatrix input) {
        //1/(1+exp(-z))
        FloatMatrix output = input.mul(-1);
        output = MatrixFunctions.expi(output);
        output = output.addi(1);
        return MatrixFunctions.powi(output, -1);
    }

    @Override
    public FloatMatrix derivate(FloatMatrix input) {
        //(1-a)*a
        return input.neg().addi(1).muli(input);
    }
}
