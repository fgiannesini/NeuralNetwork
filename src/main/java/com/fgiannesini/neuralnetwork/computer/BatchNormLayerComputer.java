package com.fgiannesini.neuralnetwork.computer;

import com.fgiannesini.neuralnetwork.computer.intermediateoutputcomputer.IntermediateOutputResult;
import com.fgiannesini.neuralnetwork.model.BatchNormLayer;
import org.jblas.DoubleMatrix;

public class BatchNormLayerComputer implements ILayerComputer<BatchNormLayer> {

    private final MeanDeviationProvider meanDeviationProvider;


    public BatchNormLayerComputer(MeanDeviationProvider meanDeviationProvider) {
        this.meanDeviationProvider = meanDeviationProvider;
    }

    public IntermediateOutputResult computeZFromInput(DoubleMatrix input, BatchNormLayer layer) {
        //Z1 = W.X
        DoubleMatrix z = layer.getWeightMatrix().mmul(input);

        MeanDeviation meanDeviation = meanDeviationProvider.get(z);

        //Z2 = (Z1 - mean) / sigma * gamma + beta
        DoubleMatrix result = z.subColumnVector(meanDeviation.getMean()).diviColumnVector(meanDeviation.getDeviation()).muliColumnVector(layer.getGammaMatrix()).addiColumnVector(layer.getBetaMatrix());
        return new IntermediateOutputResult(result, meanDeviation);
    }

}
