package com.fgiannesini.neuralnetwork.computer.data;

import org.jblas.DoubleMatrix;

import java.util.List;

public class ConvolutionData implements LayerTypeData {

    private final List<DoubleMatrix> datas;
    private int channelCount;

    public ConvolutionData(List<DoubleMatrix> datas, int channelCount) {
        this.datas = datas;
        this.channelCount = channelCount;
    }

    @Override
    public void accept(DataVisitor dataVisitor) {
        dataVisitor.visit(this);
    }

    public List<DoubleMatrix> getDatas() {
        return datas;
    }

    public int getInputCount() {
        return datas.size() / channelCount;
    }

    public int getChannelCount() {
        return channelCount;
    }
}
