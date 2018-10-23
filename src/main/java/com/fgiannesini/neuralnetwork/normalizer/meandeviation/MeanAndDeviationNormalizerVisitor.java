package com.fgiannesini.neuralnetwork.normalizer.meandeviation;

import com.fgiannesini.neuralnetwork.computer.data.*;
import org.jblas.DoubleMatrix;

import java.util.ArrayList;
import java.util.List;

public class MeanAndDeviationNormalizerVisitor implements DataVisitor {

    private final MeanDeviation meanDeviation;
    private LayerTypeData normalizedData;

    public MeanAndDeviationNormalizerVisitor(MeanDeviation meanDeviation) {
        this.meanDeviation = meanDeviation;
    }

    @Override
    public void visit(WeightBiasData data) {
        //(x-mu)/sigma
        WeightBiasMeanDeviation weightBiasMeanDeviation = (WeightBiasMeanDeviation) meanDeviation;
        DoubleMatrix normalizedMatrix = data.getData().subColumnVector(weightBiasMeanDeviation.getMean()).diviColumnVector(weightBiasMeanDeviation.getDeviation());
        normalizedData = new WeightBiasData(normalizedMatrix);
    }

    @Override
    public void visit(BatchNormData data) {
        //(x-mu)/sigma
        BatchNormMeanDeviation batchNormMeanDeviation = (BatchNormMeanDeviation) meanDeviation;
        DoubleMatrix normalizedMatrix = data.getData().subColumnVector(batchNormMeanDeviation.getMean()).diviColumnVector(batchNormMeanDeviation.getDeviation());
        normalizedData = new BatchNormData(normalizedMatrix, data.getMeanDeviationProvider());
    }

    @Override
    public void visit(ConvolutionData convolutionData) {
        //(x-mu)/sigma
        List<DoubleMatrix> datas = convolutionData.getDatas();
        int channelCount = convolutionData.getChannelCount();
        List<DoubleMatrix> results = normalizeDataList(datas, channelCount);
        normalizedData = new ConvolutionData(results, convolutionData.getChannelCount());
    }

    @Override
    public void visit(AveragePoolingData averagePoolingData) {
        //(x-mu)/sigma
        List<DoubleMatrix> datas = averagePoolingData.getDatas();
        int channelCount = averagePoolingData.getChannelCount();
        List<DoubleMatrix> results = normalizeDataList(datas, channelCount);
        normalizedData = new AveragePoolingData(results, averagePoolingData.getChannelCount());
    }

    @Override
    public void visit(MaxPoolingData maxPoolingData) {
        //(x-mu)/sigma
        List<DoubleMatrix> datas = maxPoolingData.getDatas();
        int channelCount = maxPoolingData.getChannelCount();
        List<DoubleMatrix> results = normalizeDataList(datas, channelCount);
        normalizedData = new MaxPoolingData(results, maxPoolingData.getMaxRowIndexes(), maxPoolingData.getMaxColumnIndexes(), maxPoolingData.getChannelCount());
    }

    private List<DoubleMatrix> normalizeDataList(List<DoubleMatrix> datas, int channelCount) {
        ConvolutionMeanDeviation convolutionMeanDeviation = (ConvolutionMeanDeviation) meanDeviation;
        List<DoubleMatrix> mean = convolutionMeanDeviation.getMean();
        List<DoubleMatrix> deviation = convolutionMeanDeviation.getDeviation();
        List<DoubleMatrix> results = new ArrayList<>();
        for (int i = 0; i < datas.size(); i++) {
            DoubleMatrix data = datas.get(i);
            int channelIndex = i % channelCount;
            DoubleMatrix normalizeMatrix = data.sub(mean.get(channelIndex)).divi(deviation.get(channelIndex));
            results.add(normalizeMatrix);
        }
        return results;
    }

    public LayerTypeData getNormalizedData() {
        return normalizedData;
    }
}
