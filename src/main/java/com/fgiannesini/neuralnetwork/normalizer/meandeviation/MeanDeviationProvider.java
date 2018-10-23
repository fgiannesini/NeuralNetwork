package com.fgiannesini.neuralnetwork.normalizer.meandeviation;

import com.fgiannesini.neuralnetwork.computer.data.*;
import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class MeanDeviationProvider implements DataVisitor {

    private final double epsilon = Math.pow(10, -8);
    private MeanDeviation meanDeviation;

    @Override
    public void visit(WeightBiasData data) {
        //mean
        DoubleMatrix means = data.getData().rowMeans();
        //sigma
        DoubleMatrix standardDeviation = MatrixFunctions.sqrt(MatrixFunctions.pow(data.getData().subColumnVector(means), 2).rowMeans()).addi(epsilon);
        meanDeviation = new WeightBiasMeanDeviation(means, standardDeviation);
    }

    @Override
    public void visit(BatchNormData data) {
        //mean
        DoubleMatrix means = data.getData().rowMeans();
        //sigma
        DoubleMatrix standardDeviation = MatrixFunctions.sqrt(MatrixFunctions.pow(data.getData().subColumnVector(means), 2).rowMeans()).addi(epsilon);
        meanDeviation = new BatchNormMeanDeviation(means, standardDeviation);
    }

    @Override
    public void visit(ConvolutionData convolutionData) {
        computeConvolutionMeanDeviation(convolutionData.getDatas(), convolutionData.getChannelCount(), convolutionData.getInputCount());
    }

    @Override
    public void visit(AveragePoolingData averagePoolingData) {
        computeConvolutionMeanDeviation(averagePoolingData.getDatas(), averagePoolingData.getChannelCount(), averagePoolingData.getInputCount());
    }

    @Override
    public void visit(MaxPoolingData maxPoolingData) {
        computeConvolutionMeanDeviation(maxPoolingData.getDatas(), maxPoolingData.getChannelCount(), maxPoolingData.getInputCount());
    }

    private void computeConvolutionMeanDeviation(List<DoubleMatrix> datas, int channelCount, int inputCount) {
        DoubleMatrix firstData = datas.get(0);

        List<DoubleMatrix> means = IntStream.range(0, channelCount).mapToObj(i -> DoubleMatrix.zeros(firstData.getRows(), firstData.getColumns())).collect(Collectors.toList());
        for (int i = 0; i < datas.size(); i++) {
            DoubleMatrix data = datas.get(i);
            means.get(i % channelCount).addi(data);
        }
        means.forEach(m -> m.divi(inputCount));

        //sigma
        List<DoubleMatrix> deviations = IntStream.range(0, channelCount).mapToObj(i -> DoubleMatrix.zeros(firstData.getRows(), firstData.getColumns())).collect(Collectors.toList());
        for (int i = 0; i < datas.size(); i++) {
            DoubleMatrix data = datas.get(i);
            int channelIndex = i % channelCount;
            DoubleMatrix mean = means.get(channelIndex);
            deviations.get(channelIndex).addi(MatrixFunctions.pow(data.sub(mean), 2));
        }
        deviations.forEach(m -> MatrixFunctions.sqrt(m.divi(inputCount)).addi(epsilon));
        meanDeviation = new ConvolutionMeanDeviation(means, deviations);
    }

    public MeanDeviation getMeanDeviation() {
        return meanDeviation;
    }
}
