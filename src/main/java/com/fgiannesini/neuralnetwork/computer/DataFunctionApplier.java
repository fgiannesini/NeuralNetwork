package com.fgiannesini.neuralnetwork.computer;

import org.jblas.DoubleMatrix;

import java.util.function.Function;

public class DataFunctionApplier implements DataVisitor {

    private Function<DoubleMatrix, DoubleMatrix> dataApplier;

    public DataFunctionApplier(Function<DoubleMatrix, DoubleMatrix> dataApplier) {
        this.dataApplier = dataApplier;
    }

    @Override
    public LayerTypeData visit(WeightBiasData data) {
        return new WeightBiasData(dataApplier.apply(data.getInput()));
    }

    @Override
    public LayerTypeData visit(BatchNormData data) {
        return new BatchNormData(dataApplier.apply(data.getInput()), data.getMeanDeviationProvider());
    }
}
