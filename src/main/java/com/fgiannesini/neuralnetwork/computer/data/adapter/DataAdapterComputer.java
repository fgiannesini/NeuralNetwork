package com.fgiannesini.neuralnetwork.computer.data.adapter;

import org.jblas.DoubleMatrix;
import org.jblas.ranges.IntervalRange;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

class DataAdapterComputer {

    private DataAdapterComputer() {
    }

    static DataAdapterComputer get() {
        return new DataAdapterComputer();
    }

    DoubleMatrix convertMatrixListToMatrix(List<DoubleMatrix> inputList, int inputLayerSize) {
        DoubleMatrix firstInput = inputList.get(0);
        int channelSize = firstInput.getRows() * firstInput.getColumns();
        int channelCount = inputLayerSize / channelSize;

        int inputCount = inputList.size() / channelCount;
        DoubleMatrix output = new DoubleMatrix(inputLayerSize, inputCount);
        int currentChannel = 0;
        int currentInput = 0;
        for (DoubleMatrix input : inputList) {
            DoubleMatrix inputAsRow = input.dup().reshape(input.length, 1);
            output.put(new IntervalRange(currentChannel * inputAsRow.length, (currentChannel + 1) * inputAsRow.length), new IntervalRange(currentInput, currentInput + 1), inputAsRow);
            currentChannel++;
            if (currentChannel == channelCount) {
                currentChannel = 0;
                currentInput++;
            }
        }
        return output;
    }

    List<DoubleMatrix> adaptMatrices(List<DoubleMatrix> inputs, int width, int height) {
        List<DoubleMatrix> outputs;
        int rowsCount = inputs.get(0).getRows();
        int columnsCount = inputs.get(0).getColumns();
        if (width > columnsCount || height > rowsCount) {
            outputs = inputs.stream().map(input -> {
                DoubleMatrix output = DoubleMatrix.zeros(height, width);
                output.put(new IntervalRange(0, input.getRows()), new IntervalRange(0, input.getColumns()), input);
                return output;
            }).collect(Collectors.toList());
        } else if (width < columnsCount || height < rowsCount) {
            outputs = inputs.stream().map(input -> input.get(new IntervalRange(0, width), new IntervalRange(0, height))).collect(Collectors.toList());
        } else {
            outputs = inputs;
        }
        return outputs;
    }

    List<DoubleMatrix> convertMatrixToMatrixList(DoubleMatrix input, int outputWidth, int outputHeight, int channelCount) {
        int inputSize = input.getColumns();
        int channelSize = outputHeight * outputWidth;
        List<DoubleMatrix> outputs = new ArrayList<>();
        for (int inputIndex = 0; inputIndex < inputSize; inputIndex++) {
            for (int channelIndex = 0; channelIndex < channelCount; channelIndex++) {
                DoubleMatrix output = input.get(new IntervalRange(channelIndex * channelSize, (channelIndex + 1) * channelSize), new IntervalRange(inputIndex, inputIndex + 1));
                output.reshape(outputHeight, outputWidth);
                outputs.add(output);
            }
        }
        return outputs;
    }
}
