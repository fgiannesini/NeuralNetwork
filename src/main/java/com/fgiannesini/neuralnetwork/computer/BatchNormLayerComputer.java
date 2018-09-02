package com.fgiannesini.neuralnetwork.computer;

import com.fgiannesini.neuralnetwork.computer.intermediateoutputcomputer.IntermediateOutputResult;
import com.fgiannesini.neuralnetwork.model.BatchNormLayer;
import org.jblas.DoubleMatrix;

public class BatchNormLayerComputer implements ILayerComputer<BatchNormLayer> {

    private final MeanDeviationProvider meanDeviationProvider;
    private final double epsilon = Math.pow(10, -8);

    public BatchNormLayerComputer(MeanDeviationProvider meanDeviationProvider) {
        this.meanDeviationProvider = meanDeviationProvider;
    }

    public IntermediateOutputResult computeZFromInput(DoubleMatrix input, BatchNormLayer layer) {
        //Z1 = W.X
        DoubleMatrix z = layer.getWeightMatrix().mmul(input);

        MeanDeviation meanDeviation = meanDeviationProvider.get(z);

        //Z2 = (Z1 - mean) / sigma * gamma + beta
        DoubleMatrix result = z.subColumnVector(meanDeviation.getMean()).diviColumnVector(meanDeviation.getDeviation().add(epsilon)).muliColumnVector(layer.getGammaMatrix()).addiColumnVector(layer.getBetaMatrix());
        return new IntermediateOutputResult(result, meanDeviation);
    }

}
