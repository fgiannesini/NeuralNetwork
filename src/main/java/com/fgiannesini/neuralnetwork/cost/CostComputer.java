package com.fgiannesini.neuralnetwork.cost;

import com.fgiannesini.neuralnetwork.converter.DataFormatConverter;
import org.jblas.DoubleMatrix;

public interface CostComputer {

    default double compute(double[] input, double[] output) {
        DoubleMatrix inputMatrix = DataFormatConverter.fromTabToDoubleMatrix(input);
        DoubleMatrix outputMatrix = DataFormatConverter.fromTabToDoubleMatrix(output);
        return compute(inputMatrix, outputMatrix);
    }

    default double compute(double[][] input, double[][] output) {
        DoubleMatrix inputMatrix = DataFormatConverter.fromDoubleTabToDoubleMatrix(input);
        DoubleMatrix outputMatrix = DataFormatConverter.fromDoubleTabToDoubleMatrix(output);
        return compute(inputMatrix, outputMatrix);
    }

    double compute(DoubleMatrix input, DoubleMatrix output);
}
