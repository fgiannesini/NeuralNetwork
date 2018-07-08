package com.fgiannesini.neuralnetwork.computer.finaloutputcomputer;

import org.jblas.DoubleMatrix;

public interface IFinalOutputComputer {

    default double[] compute(double[] input) {
        return compute(new DoubleMatrix(input)).toArray();
    }

    default double[][] compute(double[][] input) {
        DoubleMatrix inputMatrix = new DoubleMatrix(input).transpose();
        return compute(inputMatrix).transpose().toArray2();
    }

    DoubleMatrix compute(DoubleMatrix inputMatrix);
}
