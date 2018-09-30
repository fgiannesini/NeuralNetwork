package com.fgiannesini.neuralnetwork.computer;

import org.jblas.DoubleMatrix;

import java.util.List;

public class ConvolutionData implements LayerTypeData {

    private List<DoubleMatrix> datas;

    public ConvolutionData(List<DoubleMatrix> datas) {
        this.datas = datas;
    }

    @Override
    public void accept(DataVisitor dataVisitor) {
        dataVisitor.visit(this);
    }

    public List<DoubleMatrix> getDatas() {
        return datas;
    }
}