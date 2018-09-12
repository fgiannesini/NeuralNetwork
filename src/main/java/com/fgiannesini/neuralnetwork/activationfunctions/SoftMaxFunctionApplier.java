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
//        DoubleMatrix result = DoubleMatrix.zeros(input.getRows(), input.getColumns());
//        for (int i = 0; i < input.getColumns(); i++) {
//            DoubleMatrix column = input.getColumn(i);
//            DoubleMatrix diag = DoubleMatrix.diag(column);
//            diag.reshape(diag.length, 1);
//            DoubleMatrix derivative = diag.sub(column.mmul(column.transpose()));
//            derivative.reshape(input.getRows(), input.getRows());
//
//            result.addi(derivative.mmul(previousError));
//        }
        return previousError;
    }
}
