package com.fgiannesini.neuralnetwork.batch;

import com.fgiannesini.neuralnetwork.computer.data.*;
import org.jblas.DoubleMatrix;
import org.jblas.ranges.IntervalRange;

import java.util.List;

public class BatchIteratorVisitor implements DataVisitor {
    private final int lowerIndex;
    private final int upperIndex;
    private LayerTypeData subData;

    public BatchIteratorVisitor(int lowerIndex, int upperIndex) {
        this.lowerIndex = lowerIndex;
        this.upperIndex = upperIndex;
    }

    @Override
    public void visit(WeightBiasData data) {
        DoubleMatrix subMatrix = data.getData().getColumns(new IntervalRange(lowerIndex, upperIndex));
        subData = new WeightBiasData(subMatrix);
    }

    @Override
    public void visit(BatchNormData data) {
        DoubleMatrix subMatrix = data.getData().getColumns(new IntervalRange(lowerIndex, upperIndex));
        subData = new BatchNormData(subMatrix, data.getMeanDeviationProvider());
    }

    @Override
    public void visit(ConvolutionData convolutionData) {
        int channelCount = convolutionData.getChannelCount();
        List<DoubleMatrix> subList = convolutionData.getDatas().subList(lowerIndex * channelCount, upperIndex * channelCount);
        subData = new ConvolutionData(subList, channelCount);
    }

    @Override
    public void visit(AveragePoolingData averagePoolingData) {
        int channelCount = averagePoolingData.getChannelCount();
        List<DoubleMatrix> subList = averagePoolingData.getDatas().subList(lowerIndex * channelCount, upperIndex * channelCount);
        subData = new AveragePoolingData(subList, channelCount);
    }

    @Override
    public void visit(MaxPoolingData maxPoolingData) {
        int channelCount = maxPoolingData.getChannelCount();
        List<DoubleMatrix> subList = maxPoolingData.getDatas().subList(lowerIndex * channelCount, upperIndex * channelCount);
        subData = new MaxPoolingData(subList, maxPoolingData.getMaxRowIndexes(), maxPoolingData.getMaxColumnIndexes(), channelCount);
    }

    public LayerTypeData getSubData() {
        return subData;
    }
}
