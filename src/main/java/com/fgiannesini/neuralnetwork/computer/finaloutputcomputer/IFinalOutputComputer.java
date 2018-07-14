package com.fgiannesini.neuralnetwork.computer.finaloutputcomputer;

import com.fgiannesini.neuralnetwork.converter.DataFormatConverter;
import org.jblas.DoubleMatrix;

public interface IFinalOutputComputer {

    default double[] compute(double[] input) {
        DoubleMatrix inputMatrix = DataFormatConverter.fromTabToDoubleMatrix(input);
        return DataFormatConverter.fromDoubleMatrixToTab(compute(inputMatrix));
    }

    default double[][] compute(double[][] input) {
        DoubleMatrix inputMatrix = DataFormatConverter.fromDoubleTabToDoubleMatrix(input);
        return DataFormatConverter.fromDoubleMatrixToDoubleTab(compute(inputMatrix));
    }

    DoubleMatrix compute(DoubleMatrix inputMatrix);
}
