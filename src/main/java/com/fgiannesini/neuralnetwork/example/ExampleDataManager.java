package com.fgiannesini.neuralnetwork.example;

import com.fgiannesini.neuralnetwork.converter.DataFormatConverter;
import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

import java.util.Arrays;
import java.util.Random;

class ExampleDataManager {

    static DoubleMatrix generateInputData(int size) {
        double[] input = new Random().doubles(size).map(d -> d * 10).toArray();
        return new DoubleMatrix(1, size, input);
    }

    static DoubleMatrix convertToOutputFormat(int[] outputValues) {
        int learningSize = outputValues.length;
        double[][] output = new double[learningSize][10];
        for (int i = 0; i < learningSize; i++) {
            output[i] = new double[10];
            output[i][outputValues[i]] = 1;
        }
        return DataFormatConverter.fromDoubleTabToDoubleMatrix(output);
    }

    static DoubleMatrix convertToOutputFormat(double[] input) {
        int[] outputValues = Arrays.stream(input).map(Math::floor).mapToInt(d -> (int) d).toArray();
        return convertToOutputFormat(outputValues);
    }

    static double computeSuccessRate(DoubleMatrix expected, DoubleMatrix predicted) {
        DoubleMatrix result = convertToOutputFormat(predicted.columnArgmaxs());
        double error = MatrixFunctions.abs(result.subi(expected)).sum() / 2d;
        return (1 - error / (double) expected.columns) * 100d;
    }
}
