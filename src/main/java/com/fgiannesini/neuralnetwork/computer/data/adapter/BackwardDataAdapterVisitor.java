package com.fgiannesini.neuralnetwork.computer.data.adapter;

import com.fgiannesini.neuralnetwork.computer.data.ConvolutionData;
import com.fgiannesini.neuralnetwork.computer.data.LayerTypeData;
import com.fgiannesini.neuralnetwork.computer.data.WeightBiasData;
import com.fgiannesini.neuralnetwork.model.*;
import org.jblas.DoubleMatrix;

import java.util.List;

public class BackwardDataAdapterVisitor implements LayerVisitor {
    private final LayerTypeData previousData;
    private LayerTypeData data;

    public BackwardDataAdapterVisitor(LayerTypeData previousData) {
        this.previousData = previousData;
    }

    @Override
    public void visit(WeightBiasLayer layer) {
        if (previousData instanceof ConvolutionData) {
            List<DoubleMatrix> inputList = ((ConvolutionData) previousData).getDatas();
            DoubleMatrix output = DataAdapterComputer.get().convertMatrixListToMatrix(layer, inputList);
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
            List<DoubleMatrix> outputs = DataAdapterComputer.get().convertMatrixToMatrixList(input, layer.getOutputWidth(), layer.getOutputHeight(), layer.getChannelCount());
            data = new ConvolutionData(outputs);
        } else if (previousData instanceof ConvolutionData) {
            List<DoubleMatrix> outputs = DataAdapterComputer.get().adaptMatrices(layer.getOutputWidth(), layer.getOutputHeight(), ((ConvolutionData) previousData).getDatas());
            data = new ConvolutionData(outputs);
        }
    }

    @Override
    public void visit(MaxPoolingLayer layer) {
        if (previousData instanceof WeightBiasData) {
            DoubleMatrix input = ((WeightBiasData) previousData).getData();
            List<DoubleMatrix> outputs = DataAdapterComputer.get().convertMatrixToMatrixList(input, layer.getOutputWidth(), layer.getOutputHeight(), layer.getChannelCount());
            data = new ConvolutionData(outputs);
        } else if (previousData instanceof ConvolutionData) {
            List<DoubleMatrix> outputs = DataAdapterComputer.get().adaptMatrices(layer.getOutputWidth(), layer.getOutputHeight(), ((ConvolutionData) previousData).getDatas());
            data = new ConvolutionData(outputs);
        }
    }



    @Override
    public void visit(ConvolutionLayer layer) {
        if (previousData instanceof WeightBiasData) {
            DoubleMatrix input = ((WeightBiasData) previousData).getData();
            List<DoubleMatrix> outputs = DataAdapterComputer.get().convertMatrixToMatrixList(input, layer.getOutputWidth(), layer.getOutputHeight(), layer.getOutputChannelCount());
            data = new ConvolutionData(outputs);
        } else if (previousData instanceof ConvolutionData) {
            List<DoubleMatrix> outputs = DataAdapterComputer.get().adaptMatrices(layer.getOutputWidth(), layer.getOutputHeight(), ((ConvolutionData) previousData).getDatas());
            data = new ConvolutionData(outputs);
        }
    }

    public LayerTypeData getData() {
        return data;
    }
}
