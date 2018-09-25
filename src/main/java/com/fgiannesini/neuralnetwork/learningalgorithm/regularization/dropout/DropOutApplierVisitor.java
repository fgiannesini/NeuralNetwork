package com.fgiannesini.neuralnetwork.learningalgorithm.regularization.dropout;

import com.fgiannesini.neuralnetwork.computer.BatchNormData;
import com.fgiannesini.neuralnetwork.computer.DataVisitor;
import com.fgiannesini.neuralnetwork.computer.LayerTypeData;
import com.fgiannesini.neuralnetwork.computer.WeightBiasData;
import org.jblas.DoubleMatrix;

public class DropOutApplierVisitor implements DataVisitor {

    private DoubleMatrix dropOutMatrix;
    private LayerTypeData layerTypeData;

    public DropOutApplierVisitor(DoubleMatrix dropOutMatrix) {
        this.dropOutMatrix = dropOutMatrix;
    }

    @Override
    public void visit(WeightBiasData data) {
        layerTypeData = new WeightBiasData(data.getInput().mulColumnVector(dropOutMatrix));
    }

    @Override
    public void visit(BatchNormData data) {
        layerTypeData = new BatchNormData(data.getInput().mulColumnVector(dropOutMatrix), data.getMeanDeviationProvider());
    }

    public LayerTypeData getLayerTypeData() {
        return layerTypeData;
    }
}
