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
        DoubleMatrix inputAsVector = input.dup().reshape(input.length, 1);
        DoubleMatrix diag = DoubleMatrix.diag(input);
        diag.reshape(diag.length, 1);
        DoubleMatrix derivative = diag.sub(inputAsVector.mmul(inputAsVector.transpose()));
        derivative.reshape(input.rows, input.length);
        return derivative.mmul(previousError);
    }
}
