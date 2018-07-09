package com.fgiannesini.neuralnetwork.computer.intermediateoutputcomputer;

import org.jblas.DoubleMatrix;

import java.util.List;
import java.util.stream.Collectors;

public interface IIntermediateOutputComputer {

    default List<double[]> compute(double[] input) {
        return compute(new DoubleMatrix(input))
                .stream()
                .map(DoubleMatrix::toArray)
                .collect(Collectors.toList());
    }

    default List<double[][]> compute(double[][] input) {
        DoubleMatrix inputMatrix = new DoubleMatrix(input).transpose();
        return compute(inputMatrix)
                .stream()
                .map(result -> result.transpose().toArray2())
                .collect(Collectors.toList());
    }

    List<DoubleMatrix> compute(DoubleMatrix inputMatrix);
}
