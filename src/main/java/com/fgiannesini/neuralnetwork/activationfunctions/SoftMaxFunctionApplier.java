package com.fgiannesini.neuralnetwork.activationfunctions;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

public class SoftMaxFunctionApplier implements ActivationFunctionApplier {

    private double epsilon;

    public SoftMaxFunctionApplier() {
        epsilon = Math.pow(10, -18);
    }

    @Override
    public DoubleMatrix apply(DoubleMatrix input) {
        //exp(z)/sum(exp(z))
        double max = input.max();
        DoubleMatrix output = MatrixFunctions.exp(input.sub(max));
        DoubleMatrix sum = output.columnSums().addi(epsilon);
        return output.diviRowVector(sum);
    }

    @Override
    public DoubleMatrix derivate(DoubleMatrix input) {
        return DoubleMatrix.ones(input.getRows(), input.getColumns());
    }
}
