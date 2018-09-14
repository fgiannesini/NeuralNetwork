package com.fgiannesini.neuralnetwork.computer.finaloutputcomputer;

import com.fgiannesini.neuralnetwork.converter.DataFormatConverter;
import com.fgiannesini.neuralnetwork.model.Layer;
import org.jblas.DoubleMatrix;

public interface IFinalOutputComputer<L extends Layer> {

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
