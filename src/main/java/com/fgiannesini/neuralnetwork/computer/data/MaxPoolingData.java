package com.fgiannesini.neuralnetwork.computer.data;

import org.jblas.DoubleMatrix;

import java.util.List;

public class MaxPoolingData implements LayerTypeData {

    private final List<DoubleMatrix> datas;
    private List<DoubleMatrix> maxXIndexes;
    private List<DoubleMatrix> maxYIndexes;

    public MaxPoolingData(List<DoubleMatrix> datas, List<DoubleMatrix> maxXIndexes, List<DoubleMatrix> maxYIndexes) {
        this.datas = datas;
        this.maxXIndexes = maxXIndexes;
        this.maxYIndexes = maxYIndexes;
    }

    @Override
    public void accept(DataVisitor dataVisitor) {
        dataVisitor.visit(this);
    }

    public List<DoubleMatrix> getDatas() {
        return datas;
    }

    public List<DoubleMatrix> getMaxXIndexes() {
        return maxXIndexes;
    }

    public List<DoubleMatrix> getMaxYIndexes() {
        return maxYIndexes;
    }
}
