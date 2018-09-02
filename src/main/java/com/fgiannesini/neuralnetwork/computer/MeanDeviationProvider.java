package com.fgiannesini.neuralnetwork.computer;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

public class MeanDeviationProvider {

    public MeanDeviation get(DoubleMatrix input) {
        //mean
        DoubleMatrix means = input.rowMeans();
        //sigma
        DoubleMatrix standardDeviation = MatrixFunctions.sqrt(MatrixFunctions.pow(input.subColumnVector(means), 2).rowMeans());
        return new MeanDeviation(means, standardDeviation);
    }
}
