package com.fgiannesini.neuralnetwork.computer.data.adapter;

import com.fgiannesini.neuralnetwork.computer.data.*;
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
        } else if (previousData instanceof AveragePoolingData) {
            List<DoubleMatrix> inputList = ((AveragePoolingData) previousData).getDatas();
            DoubleMatrix output = DataAdapterComputer.get().convertMatrixListToMatrix(layer, inputList);
            data = new WeightBiasData(output);
        } else if (previousData instanceof MaxPoolingData) {
            List<DoubleMatrix> inputList = ((MaxPoolingData) previousData).getDatas();
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
            data = new AveragePoolingData(outputs, layer.getChannelCount());
        } else if (previousData instanceof ConvolutionData) {
            List<DoubleMatrix> outputs = DataAdapterComputer.get().adaptMatrices(layer.getOutputWidth(), layer.getOutputHeight(), ((ConvolutionData) previousData).getDatas());
            data = new AveragePoolingData(outputs, layer.getChannelCount());
        } else if (previousData instanceof AveragePoolingData) {
            List<DoubleMatrix> outputs = DataAdapterComputer.get().adaptMatrices(layer.getOutputWidth(), layer.getOutputHeight(), ((AveragePoolingData) previousData).getDatas());
            data = new AveragePoolingData(outputs, layer.getChannelCount());
        } else if (previousData instanceof MaxPoolingData) {
            MaxPoolingData maxPoolingData = (MaxPoolingData) this.previousData;
            List<DoubleMatrix> outputs = DataAdapterComputer.get().adaptMatrices(layer.getOutputWidth(), layer.getOutputHeight(), maxPoolingData.getDatas());
            data = new AveragePoolingData(outputs, layer.getChannelCount());
        }
    }

    @Override
    public void visit(MaxPoolingLayer layer) {
        if (previousData instanceof WeightBiasData) {
            DoubleMatrix input = ((WeightBiasData) previousData).getData();
            List<DoubleMatrix> outputs = DataAdapterComputer.get().convertMatrixToMatrixList(input, layer.getOutputWidth(), layer.getOutputHeight(), layer.getChannelCount());
            data = new MaxPoolingData(outputs, null, null, layer.getChannelCount());
        } else if (previousData instanceof ConvolutionData) {
            List<DoubleMatrix> outputs = DataAdapterComputer.get().adaptMatrices(layer.getOutputWidth(), layer.getOutputHeight(), ((ConvolutionData) previousData).getDatas());
            data = new MaxPoolingData(outputs, null, null, layer.getChannelCount());
        } else if (previousData instanceof AveragePoolingData) {
            List<DoubleMatrix> outputs = DataAdapterComputer.get().adaptMatrices(layer.getOutputWidth(), layer.getOutputHeight(), ((AveragePoolingData) previousData).getDatas());
            data = new MaxPoolingData(outputs, null, null, layer.getChannelCount());
        } else if (previousData instanceof MaxPoolingData) {
            MaxPoolingData maxPoolingData = (MaxPoolingData) this.previousData;
            List<DoubleMatrix> outputs = DataAdapterComputer.get().adaptMatrices(layer.getOutputWidth(), layer.getOutputHeight(), maxPoolingData.getDatas());
            data = new MaxPoolingData(outputs, maxPoolingData.getMaxRowIndexes(), null, layer.getChannelCount());
        }
    }



    @Override
    public void visit(ConvolutionLayer layer) {
        if (previousData instanceof WeightBiasData) {
            DoubleMatrix input = ((WeightBiasData) previousData).getData();
            List<DoubleMatrix> outputs = DataAdapterComputer.get().convertMatrixToMatrixList(input, layer.getOutputWidth(), layer.getOutputHeight(), layer.getOutputChannelCount());
            data = new ConvolutionData(outputs, layer.getOutputChannelCount());
        } else if (previousData instanceof ConvolutionData) {
            List<DoubleMatrix> outputs = DataAdapterComputer.get().adaptMatrices(layer.getOutputWidth(), layer.getOutputHeight(), ((ConvolutionData) previousData).getDatas());
            data = new ConvolutionData(outputs, layer.getOutputChannelCount());
        } else if (previousData instanceof AveragePoolingData) {
            List<DoubleMatrix> outputs = DataAdapterComputer.get().adaptMatrices(layer.getOutputWidth(), layer.getOutputHeight(), ((AveragePoolingData) previousData).getDatas());
            data = new ConvolutionData(outputs, layer.getOutputChannelCount());
        } else if (previousData instanceof MaxPoolingData) {
            MaxPoolingData maxPoolingData = (MaxPoolingData) this.previousData;
            List<DoubleMatrix> outputs = DataAdapterComputer.get().adaptMatrices(layer.getOutputWidth(), layer.getOutputHeight(), maxPoolingData.getDatas());
            data = new ConvolutionData(outputs, layer.getOutputChannelCount());
        }
    }

    public LayerTypeData getData() {
        return data;
    }
}
