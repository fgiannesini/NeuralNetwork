package com.fgiannesini.neuralnetwork.math;

import org.jblas.DoubleMatrix;
import org.jblas.ranges.IntervalRange;

import java.util.function.Function;

public class ConvolutionComputer {

    private ConvolutionComputer() {
    }

    public static ConvolutionComputer get() {
        return new ConvolutionComputer();
    }

    public DoubleMatrix computeConvolution(DoubleMatrix input, Function<DoubleMatrix, Double> convolutionApplication, int padding, int stride, int filterSize) {
        DoubleMatrix paddedInput = DoubleMatrix.zeros(input.rows + 2 * padding, input.columns + 2 * padding);
        paddedInput.put(new IntervalRange(padding, paddedInput.getRows() - padding), new IntervalRange(padding, paddedInput.getColumns() - padding), input);

        int outputRowCount = (int) Math.ceil((input.getRows() + 2 * padding - filterSize) / (double) stride + 1);
        int outputColumnCount = (int) Math.ceil((input.getColumns() + 2 * padding - filterSize) / (double) stride + 1);
        DoubleMatrix output = DoubleMatrix.zeros(outputRowCount, outputColumnCount);
        for (int rowIndex = 0; rowIndex < paddedInput.getRows() - filterSize + 1; rowIndex += stride) {
            for (int columnIndex = 0; columnIndex < paddedInput.getColumns() - filterSize + 1; columnIndex += stride) {
                DoubleMatrix inputPart = paddedInput.get(new IntervalRange(rowIndex, rowIndex + filterSize), new IntervalRange(columnIndex, columnIndex + filterSize));
                output.put(rowIndex / stride, columnIndex / stride, convolutionApplication.apply(inputPart));
            }
        }

        return output;
    }
}
