package com.fgiannesini.neuralnetwork.activationfunctions;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

public class SoftMaxFunctionApplier implements ActivationFunctionApplier {

    @Override
    public DoubleMatrix apply(DoubleMatrix input) {
        //exp(z)/sum(exp(z))
        DoubleMatrix output = MatrixFunctions.exp(input);
        DoubleMatrix sum = output.columnSums();
        return output.diviRowVector(sum);
    }

    @Override
    public DoubleMatrix derivate(DoubleMatrix input) {
        return DoubleMatrix.ones(input.getRows(), input.getColumns());
    }
}
