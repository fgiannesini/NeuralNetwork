package com.fgiannesini.neuralnetwork.computer.data;

import com.fgiannesini.neuralnetwork.normalizer.meandeviation.MeanDeviationProvider;
import org.jblas.DoubleMatrix;

public class BatchNormData implements LayerTypeData {

    private final DoubleMatrix input;
    private final MeanDeviationProvider meanDeviationProvider;

    public BatchNormData(DoubleMatrix input, MeanDeviationProvider meanDeviationProvider) {
        this.input = input;
        this.meanDeviationProvider = meanDeviationProvider;
    }

    public DoubleMatrix getData() {
        return input;
    }

    public MeanDeviationProvider getMeanDeviationProvider() {
        return meanDeviationProvider;
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
