package com.fgiannesini.neuralnetwork.computer.intermediateoutputcomputer;

import com.fgiannesini.neuralnetwork.converter.DataFormatConverter;
import org.jblas.DoubleMatrix;

import java.util.List;
import java.util.stream.Collectors;

public interface IIntermediateOutputComputer {

    default List<double[]> compute(double[] input) {
        DoubleMatrix inputMatrix = DataFormatConverter.fromTabToDoubleMatrix(input);
        return compute(inputMatrix).stream().map(DataFormatConverter::fromDoubleMatrixToTab)
                .collect(Collectors.toList());
    }

    default List<double[][]> compute(double[][] input) {
        DoubleMatrix inputMatrix = DataFormatConverter.fromDoubleTabToDoubleMatrix(input);
        return compute(inputMatrix).stream().map(DataFormatConverter::fromDoubleMatrixToDoubleTab)
                .collect(Collectors.toList());
    }

    List<DoubleMatrix> compute(DoubleMatrix inputMatrix);
}
