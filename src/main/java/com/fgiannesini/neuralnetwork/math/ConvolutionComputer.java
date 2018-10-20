package com.fgiannesini.neuralnetwork.math;

import org.jblas.DoubleMatrix;
import org.jblas.ranges.IntervalRange;

import java.util.function.BiFunction;

public class ConvolutionComputer {

    private ConvolutionComputer() {
    }

    public static ConvolutionComputer get() {
        return new ConvolutionComputer();
    }

    public DoubleMatrix computeConvolution(DoubleMatrix input, BiFunction<DoubleMatrix, ConvCoords, Double> convolutionApplication, int padding, int stride, int filterSize) {
        DoubleMatrix paddedInput = applyPadding(input, padding);

        int outputRowCount = computeOutputSize(padding, stride, filterSize, input.getRows());
        int outputColumnCount = computeOutputSize(padding, stride, filterSize, input.getColumns());
        DoubleMatrix output = DoubleMatrix.zeros(outputRowCount, outputColumnCount);
        for (int rowIndex = 0; rowIndex < paddedInput.getRows() - filterSize + 1; rowIndex += stride) {
            for (int columnIndex = 0; columnIndex < paddedInput.getColumns() - filterSize + 1; columnIndex += stride) {
                DoubleMatrix inputPart = paddedInput.get(new IntervalRange(rowIndex, rowIndex + filterSize), new IntervalRange(columnIndex, columnIndex + filterSize));
                ConvCoords coords = new ConvCoords(rowIndex, columnIndex);
                output.put(rowIndex / stride, columnIndex / stride, convolutionApplication.apply(inputPart, coords));
            }
        }

        return output;
    }

    public DoubleMatrix applyPadding(DoubleMatrix input, int padding) {
        if (padding != 0) {
            DoubleMatrix paddedInput = DoubleMatrix.zeros(input.rows + 2 * padding, input.columns + 2 * padding);
            paddedInput.put(new IntervalRange(padding, paddedInput.getRows() - padding), new IntervalRange(padding, paddedInput.getColumns() - padding), input);
            return paddedInput;
        }
        return input;
    }

    public DoubleMatrix removePadding(DoubleMatrix input, int padding) {
        if (padding != 0) {
            return input.get(new IntervalRange(padding, input.getRows() - padding), new IntervalRange(padding, input.getColumns() - padding));
        }
        return input;
    }

    public int computeOutputSize(int padding, int stride, int filterSize, int rows) {
        return (rows + 2 * padding - filterSize) / stride + 1;
    }
}
