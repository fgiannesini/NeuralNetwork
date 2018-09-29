package com.fgiannesini.neuralnetwork.computer;

import org.jblas.DoubleMatrix;

import java.util.function.Function;

public class DataFunctionApplier implements DataVisitor {

    private final Function<DoubleMatrix, DoubleMatrix> dataApplier;
    private LayerTypeData layerTypeData;

    public DataFunctionApplier(Function<DoubleMatrix, DoubleMatrix> dataApplier) {
        this.dataApplier = dataApplier;
    }

    @Override
    public void visit(WeightBiasData data) {
        layerTypeData = new WeightBiasData(dataApplier.apply(data.getInput()));
    }

    @Override
    public void visit(BatchNormData data) {
        layerTypeData = new BatchNormData(dataApplier.apply(data.getInput()), data.getMeanDeviationProvider());
    }

    public LayerTypeData getLayerTypeData() {
        return layerTypeData;
    }
}
