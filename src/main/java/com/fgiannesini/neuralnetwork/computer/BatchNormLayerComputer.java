package com.fgiannesini.neuralnetwork.computer;

import com.fgiannesini.neuralnetwork.model.BatchNormLayer;
import org.jblas.DoubleMatrix;

import java.util.function.Function;

public class BatchNormLayerComputer implements ILayerComputer<BatchNormLayer> {

    private final Function<DoubleMatrix, MeanDeviation> meanDeviationProvider;

    public BatchNormLayerComputer(Function<DoubleMatrix, MeanDeviation> meanDeviationProvider) {
        this.meanDeviationProvider = meanDeviationProvider;
    }

    public DoubleMatrix computeZFromInput(DoubleMatrix input, BatchNormLayer layer) {
        //Z1 = W.X
        DoubleMatrix z = layer.getWeightMatrix().mmul(input);

        MeanDeviation meanDeviation = meanDeviationProvider.apply(z);

        //Z2 = (Z1 - mean) / sigma * gamma + beta
        return z.subColumnVector(meanDeviation.getMean()).diviColumnVector(meanDeviation.getDeviation()).muliColumnVector(layer.getGammaMatrix()).addiColumnVector(layer.getBetaMatrix());
    }

}
