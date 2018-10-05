package com.fgiannesini.neuralnetwork.computer.intermediateoutputcomputer;

import com.fgiannesini.neuralnetwork.computer.data.ConvolutionData;
import com.fgiannesini.neuralnetwork.computer.data.LayerTypeData;
import com.fgiannesini.neuralnetwork.computer.data.WeightBiasData;
import com.fgiannesini.neuralnetwork.model.*;
import org.jblas.DoubleMatrix;
import org.jblas.ranges.IntervalRange;

import java.util.ArrayList;
import java.util.List;

public class DataAdapterVisitor implements LayerVisitor {
    private final LayerTypeData previousData;
    private LayerTypeData data;

    public DataAdapterVisitor(LayerTypeData previousData) {
        this.previousData = previousData;
    }

    @Override
    public void visit(WeightBiasLayer layer) {
        if (previousData instanceof ConvolutionData) {
            List<DoubleMatrix> inputList = ((ConvolutionData) previousData).getDatas();

            DoubleMatrix firstInput = inputList.get(0);
            int channelSize = firstInput.getRows() * firstInput.getColumns();
            int channelCount = layer.getInputLayerSize() / channelSize;

            int inputCount = inputList.size() / channelCount;
            DoubleMatrix output = new DoubleMatrix(layer.getInputLayerSize(), inputCount);
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
            data = new WeightBiasData(output);
        } else {
            data = previousData;
        }
    }

    @Override
    public void visit(BatchNormLayer layer) {
        data = previousData;
    }

    @Override
    public void visit(AveragePoolingLayer layer) {
        if (previousData instanceof WeightBiasData) {
            DoubleMatrix input = ((WeightBiasData) previousData).getData();
            List<DoubleMatrix> outputs = convertMatrixToMatrixList(input, layer.getOutputWidth(), layer.getOutputHeight(), layer.getChannelCount());
            data = new ConvolutionData(outputs);
        } else {
            data = previousData;
        }
    }

    @Override
    public void visit(MaxPoolingLayer layer) {
        if (previousData instanceof WeightBiasData) {
            DoubleMatrix input = ((WeightBiasData) previousData).getData();
            List<DoubleMatrix> outputs = convertMatrixToMatrixList(input, layer.getOutputWidth(), layer.getOutputHeight(), layer.getChannelCount());
            data = new ConvolutionData(outputs);
        } else {
            data = previousData;
        }
    }

    private List<DoubleMatrix> convertMatrixToMatrixList(DoubleMatrix input, int outputWidth, int outputHeight, int channelCount) {
        int inputSize = input.getColumns();
        int channelSize = outputHeight * outputWidth;
        List<DoubleMatrix> outputs = new ArrayList<>();
        for (int inputIndex = 0; inputIndex < inputSize; inputIndex++) {
            for (int channelIndex = 0; channelIndex < channelCount; channelIndex++) {
                DoubleMatrix output = input.get(new IntervalRange(channelIndex * channelSize, (channelIndex + 1) * channelSize), new IntervalRange(inputIndex, inputIndex + 1));
                output.reshape(outputWidth, outputHeight);
                outputs.add(output);
            }
        }
        return outputs;
    }

    @Override
    public void visit(ConvolutionLayer layer) {
        if (previousData instanceof WeightBiasData) {
            DoubleMatrix input = ((WeightBiasData) previousData).getData();
            List<DoubleMatrix> outputs = convertMatrixToMatrixList(input, layer.getOutputWidth(), layer.getOutputHeight(), layer.getOutputChannelCount());
            data = new ConvolutionData(outputs);
        } else {
            data = previousData;
        }
    }

    public LayerTypeData getData() {
        return data;
    }
}
