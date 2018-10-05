package com.fgiannesini.neuralnetwork.learningalgorithm.regularization.dropout;

import com.fgiannesini.neuralnetwork.computer.data.BatchNormData;
import com.fgiannesini.neuralnetwork.computer.data.DataVisitor;
import com.fgiannesini.neuralnetwork.computer.data.LayerTypeData;
import com.fgiannesini.neuralnetwork.computer.data.WeightBiasData;
import org.jblas.DoubleMatrix;

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

    public LayerTypeData getLayerTypeData() {
        return layerTypeData;
    }
}
