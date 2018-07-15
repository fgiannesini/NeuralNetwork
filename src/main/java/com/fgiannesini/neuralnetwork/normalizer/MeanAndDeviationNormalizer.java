package com.fgiannesini.neuralnetwork.normalizer;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

public class MeanAndDeviationNormalizer implements INormalizer {

    @Override
    public DoubleMatrix normalize(DoubleMatrix input) {
        //mu
        DoubleMatrix means = input.rowMeans();
        //sigma
        DoubleMatrix standardDeviation = MatrixFunctions.sqrt(MatrixFunctions.pow(input, 2).rowMeans());
        standardDeviation = standardDeviation.not().addi(standardDeviation);
        //(x-mu)/sigma
        DoubleMatrix output = input.subColumnVector(means).diviColumnVector(standardDeviation);
        return output;
    }
}
