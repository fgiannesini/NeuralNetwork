package com.fgiannesini.neuralnetwork.normalizer.meandeviation;

import com.fgiannesini.neuralnetwork.computer.data.BatchNormData;
import com.fgiannesini.neuralnetwork.computer.data.DataVisitor;
import com.fgiannesini.neuralnetwork.computer.data.LayerTypeData;
import com.fgiannesini.neuralnetwork.computer.data.WeightBiasData;
import org.jblas.DoubleMatrix;

public class MeanAndDeviationNormalizerVisitor implements DataVisitor {

    private final MeanDeviation meanDeviation;
    private LayerTypeData normalizedData;

    public MeanAndDeviationNormalizerVisitor(MeanDeviation meanDeviation) {
        this.meanDeviation = meanDeviation;
    }

    @Override
    public void visit(WeightBiasData data) {
        DoubleMatrix normalizedMatrix = normalizeInput(data.getData());
        normalizedData = new WeightBiasData(normalizedMatrix);
    }

    @Override
    public void visit(BatchNormData data) {
        DoubleMatrix normalizedMatrix = normalizeInput(data.getData());
        normalizedData = new BatchNormData(normalizedMatrix, data.getMeanDeviationProvider());
    }

    private DoubleMatrix normalizeInput(DoubleMatrix data) {
        //(x-mu)/sigma
        return data.subColumnVector(meanDeviation.getMean()).diviColumnVector(meanDeviation.getDeviation());
    }

    public LayerTypeData getNormalizedData() {
        return normalizedData;
    }
}
