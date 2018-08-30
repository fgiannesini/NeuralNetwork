package com.fgiannesini.neuralnetwork.computer;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

class ComputationUtils {
    private static final double epsilon = Math.pow(10, -8);

    static MeanDeviation get(DoubleMatrix input) {
        //mean
        DoubleMatrix means = input.rowMeans();
        //sigma
        DoubleMatrix standardDeviation = MatrixFunctions.sqrt(MatrixFunctions.pow(input.subColumnVector(means), 2).rowMeans().addi(epsilon));
        return new MeanDeviation(means, standardDeviation);
    }
}
