package com.fgiannesini.neuralnetwork.computer.data;

import org.jblas.DoubleMatrix;

import java.util.List;

public class AveragePoolingData implements LayerTypeData {

    private final List<DoubleMatrix> datas;

    public AveragePoolingData(List<DoubleMatrix> datas) {
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
