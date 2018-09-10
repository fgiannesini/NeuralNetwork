package com.fgiannesini.neuralnetwork.activationfunctions;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

public class SoftMaxFunctionApplier implements ActivationFunctionApplier {

    @Override
    public DoubleMatrix apply(DoubleMatrix input) {
        //exp(z)/sum(exp(z))
        DoubleMatrix max = input.columnMaxs();
        DoubleMatrix output = MatrixFunctions.exp(input.subRowVector(max));
        DoubleMatrix sum = output.columnSums();
        return output.diviRowVector(sum);
    }

    @Override
    public DoubleMatrix derivate(DoubleMatrix input, DoubleMatrix previousError) {
        DoubleMatrix derivative = DoubleMatrix.eye(input.getRows()).mulColumnVector(input).sub(input.mmul(input.transpose()));
        return derivative.mmul(previousError);
    }
}
