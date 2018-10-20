package com.fgiannesini.neuralnetwork.computer.data;

import org.jblas.DoubleMatrix;

import java.util.List;

public class MaxPoolingData implements LayerTypeData {

    private final List<DoubleMatrix> datas;
    private final List<DoubleMatrix> maxRowIndexes;
    private final List<DoubleMatrix> maxColumnIndexes;

    public MaxPoolingData(List<DoubleMatrix> datas, List<DoubleMatrix> maxRowIndexes, List<DoubleMatrix> maxColumnIndexes) {
        this.datas = datas;
        this.maxRowIndexes = maxRowIndexes;
        this.maxColumnIndexes = maxColumnIndexes;
    }

    @Override
    public void accept(DataVisitor dataVisitor) {
        dataVisitor.visit(this);
    }

    public List<DoubleMatrix> getDatas() {
        return datas;
    }

    public List<DoubleMatrix> getMaxRowIndexes() {
        return maxRowIndexes;
    }

    public List<DoubleMatrix> getMaxColumnIndexes() {
        return maxColumnIndexes;
    }
}
