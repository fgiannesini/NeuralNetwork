package com.fgiannesini.neuralnetwork.computer.intermediateoutputcomputer;

import com.fgiannesini.neuralnetwork.converter.DataFormatConverter;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.GradientLayerProvider;
import org.jblas.DoubleMatrix;

public interface IIntermediateOutputComputer {

    default GradientLayerProvider compute(double[] input) {
        DoubleMatrix inputMatrix = DataFormatConverter.fromTabToDoubleMatrix(input);
        return compute(inputMatrix);
    }

    default GradientLayerProvider compute(double[][] input) {
        DoubleMatrix inputMatrix = DataFormatConverter.fromDoubleTabToDoubleMatrix(input);
        return compute(inputMatrix);
    }

    GradientLayerProvider compute(DoubleMatrix inputMatrix);
}
