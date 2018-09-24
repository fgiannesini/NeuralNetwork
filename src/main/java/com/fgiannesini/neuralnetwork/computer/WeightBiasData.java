package com.fgiannesini.neuralnetwork.computer;

import org.jblas.DoubleMatrix;

public class WeightBiasData implements LayerTypeData {

    private DoubleMatrix input;

    public WeightBiasData(DoubleMatrix input) {
        this.input = input;
    }

    public DoubleMatrix getInput() {
        return input;
    }

    @Override
    public void accept(DataVisitor dataVisitor) {
        dataVisitor.visit(this);
    }
}
