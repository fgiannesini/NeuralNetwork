package com.fgiannesini.neuralnetwork.converter;

import org.jblas.DoubleMatrix;

public class DataFormatConverter {

    public static DoubleMatrix fromTabToDoubleMatrix(double[] input) {
        return new DoubleMatrix(input);
    }

    public static DoubleMatrix fromDoubleTabToDoubleMatrix(double[][] input) {
        return new DoubleMatrix(input).transpose();
    }

    public static double[] fromDoubleMatrixToTab(DoubleMatrix input) {
        return input.toArray();
    }

    public static double[][] fromDoubleMatrixToDoubleTab(DoubleMatrix input) {
        return input.transpose().toArray2();
    }

}
