package com.fgiannesini.neuralnetwork.computer.intermediateoutputcomputer;

import com.fgiannesini.neuralnetwork.computer.ConvolutionData;
import com.fgiannesini.neuralnetwork.computer.LayerTypeData;
import com.fgiannesini.neuralnetwork.computer.WeightBiasData;
import com.fgiannesini.neuralnetwork.model.*;
import org.jblas.DoubleMatrix;
import org.jblas.ranges.IntervalRange;

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
        data = previousData;
    }

    @Override
    public void visit(MaxPoolingLayer layer) {
        data = previousData;
    }

    @Override
    public void visit(ConvolutionLayer layer) {
        data = previousData;
    }

    public LayerTypeData getData() {
        return data;
    }
}
