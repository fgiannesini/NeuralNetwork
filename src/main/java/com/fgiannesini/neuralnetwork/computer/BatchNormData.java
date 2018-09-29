package com.fgiannesini.neuralnetwork.computer;

import org.jblas.DoubleMatrix;

public class BatchNormData implements LayerTypeData {

    private final DoubleMatrix input;
    private final MeanDeviationProvider meanDeviationProvider;

    public BatchNormData(DoubleMatrix input, MeanDeviationProvider meanDeviationProvider) {
        this.input = input;
        this.meanDeviationProvider = meanDeviationProvider;
    }

    public DoubleMatrix getInput() {
        return input;
    }

    public MeanDeviationProvider getMeanDeviationProvider() {
        return meanDeviationProvider;
    }

    @Override
    public void accept(DataVisitor dataVisitor) {
        dataVisitor.visit(this);
    }
}
