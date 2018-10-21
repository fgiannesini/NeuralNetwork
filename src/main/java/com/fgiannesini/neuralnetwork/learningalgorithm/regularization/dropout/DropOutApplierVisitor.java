package com.fgiannesini.neuralnetwork.learningalgorithm.regularization.dropout;

import com.fgiannesini.neuralnetwork.computer.data.*;
import org.jblas.DoubleMatrix;

import java.util.List;
import java.util.stream.Collectors;

public class DropOutApplierVisitor implements DataVisitor {

    private final DoubleMatrix dropOutMatrix;
    private LayerTypeData layerTypeData;

    public DropOutApplierVisitor(DoubleMatrix dropOutMatrix) {
        this.dropOutMatrix = dropOutMatrix;
    }

    @Override
    public void visit(WeightBiasData data) {
        layerTypeData = new WeightBiasData(data.getData().mulColumnVector(dropOutMatrix));
    }

    @Override
    public void visit(BatchNormData data) {
        layerTypeData = new BatchNormData(data.getData().mulColumnVector(dropOutMatrix), data.getMeanDeviationProvider());
    }

    @Override
    public void visit(ConvolutionData data) {
        List<DoubleMatrix> result = data.getDatas().stream()
                .map(matrix -> matrix.mul(dropOutMatrix))
                .collect(Collectors.toList());
        layerTypeData = new ConvolutionData(result, data.getChannelCount());
    }

    @Override
    public void visit(AveragePoolingData averagePoolingData) {
        layerTypeData = averagePoolingData;
    }

    @Override
    public void visit(MaxPoolingData maxPoolingData) {
        layerTypeData = maxPoolingData;
    }

    public LayerTypeData getLayerTypeData() {
        return layerTypeData;
    }
}
