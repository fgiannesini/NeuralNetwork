package com.fgiannesini.neuralnetwork.computer.data;

import org.jblas.DoubleMatrix;

import java.util.List;

public class MaxPoolingData implements LayerTypeData {

    private final List<DoubleMatrix> datas;
    private List<DoubleMatrix> maxIndexes;

    public MaxPoolingData(List<DoubleMatrix> datas, List<DoubleMatrix> maxIndexes) {
        this.datas = datas;
        this.maxIndexes = maxIndexes;
    }

    @Override
    public void accept(DataVisitor dataVisitor) {
        dataVisitor.visit(this);
    }

    public List<DoubleMatrix> getDatas() {
        return datas;
    }

    public List<DoubleMatrix> getMaxIndexes() {
        return maxIndexes;
    }
}
