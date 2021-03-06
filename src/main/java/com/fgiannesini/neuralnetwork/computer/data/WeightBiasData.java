package com.fgiannesini.neuralnetwork.computer.data;

import org.jblas.DoubleMatrix;

public class WeightBiasData implements LayerTypeData {

    private final DoubleMatrix input;

    public WeightBiasData(DoubleMatrix input) {
        this.input = input;
    }

    public DoubleMatrix getData() {
        return input;
    }

    @Override
    public void accept(DataVisitor dataVisitor) {
        dataVisitor.visit(this);
    }

    @Override
    public int getInputCount() {
        return input.getColumns();
    }
}
