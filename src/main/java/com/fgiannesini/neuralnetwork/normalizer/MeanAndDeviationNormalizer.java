package com.fgiannesini.neuralnetwork.normalizer;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

public class MeanAndDeviationNormalizer implements INormalizer {

    private DoubleMatrix means;
    private DoubleMatrix standardDeviation;

    @Override
    public DoubleMatrix normalize(DoubleMatrix input) {
        if (means == null || standardDeviation == null) {
            //mu
            means = input.rowMeans();

            //sigma
            standardDeviation = MatrixFunctions.sqrt(MatrixFunctions.pow(input, 2).rowMeans());
            standardDeviation = standardDeviation.not().addi(standardDeviation);
        }
        //(x-mu)/sigma
        return input.subColumnVector(means).diviColumnVector(standardDeviation);
    }


}
