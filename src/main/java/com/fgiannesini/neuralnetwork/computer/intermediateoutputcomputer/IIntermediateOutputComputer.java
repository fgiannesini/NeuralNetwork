package com.fgiannesini.neuralnetwork.computer.intermediateoutputcomputer;

import com.fgiannesini.neuralnetwork.converter.DataFormatConverter;
import com.fgiannesini.neuralnetwork.model.Layer;
import org.jblas.DoubleMatrix;

import java.util.List;
import java.util.stream.Collectors;

public interface IIntermediateOutputComputer<L extends Layer> {

    default List<double[]> compute(double[] input) {
        DoubleMatrix inputMatrix = DataFormatConverter.fromTabToDoubleMatrix(input);
        return compute(inputMatrix).stream()
                .map(IntermediateOutputResult::getResult)
                .map(DataFormatConverter::fromDoubleMatrixToTab)
                .collect(Collectors.toList());
    }

    default List<double[][]> compute(double[][] input) {
        DoubleMatrix inputMatrix = DataFormatConverter.fromDoubleTabToDoubleMatrix(input);
        return compute(inputMatrix).stream()
                .map(IntermediateOutputResult::getResult)
                .map(DataFormatConverter::fromDoubleMatrixToDoubleTab)
                .collect(Collectors.toList());
    }

    List<IntermediateOutputResult> compute(DoubleMatrix inputMatrix);
}
