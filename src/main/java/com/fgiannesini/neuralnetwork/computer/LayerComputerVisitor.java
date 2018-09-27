package com.fgiannesini.neuralnetwork.computer;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionApplier;
import com.fgiannesini.neuralnetwork.computer.intermediateoutputcomputer.IntermediateOutputResult;
import com.fgiannesini.neuralnetwork.model.*;
import org.jblas.DoubleMatrix;

public class LayerComputerVisitor implements LayerVisitor {

    private LayerTypeData layerTypeData;
    private IntermediateOutputResult intermediateOutputResult;

    public LayerComputerVisitor(LayerTypeData layerTypeData) {
        this.layerTypeData = layerTypeData;
    }

    @Override
    public void visit(WeightBiasLayer layer) {
        //W.X + b
        WeightBiasData weightBiasData = (WeightBiasData) layerTypeData;
        DoubleMatrix result = layer.getWeightMatrix().mmul(weightBiasData.getInput()).addiColumnVector(layer.getBiasMatrix());
        ActivationFunctionApplier activationFunctionApplier = layer.getActivationFunctionType().getActivationFunction();
        DoubleMatrix activatedResult = activationFunctionApplier.apply(result);
        intermediateOutputResult = new IntermediateOutputResult(new WeightBiasData(activatedResult));
    }

    @Override
    public void visit(BatchNormLayer layer) {
        //Z1 = W.X
        BatchNormData batchNormData = (BatchNormData) layerTypeData;
        DoubleMatrix z = layer.getWeightMatrix().mmul(batchNormData.getInput());

        MeanDeviationProvider meanDeviationProvider = batchNormData.getMeanDeviationProvider();
        new BatchNormData(z, null).accept(meanDeviationProvider);
        MeanDeviation meanDeviation = meanDeviationProvider.getMeanDeviation();

        //Z2 = (Z1 - mean) / sigma * gamma + beta
        DoubleMatrix afterMeanApplicationResult = z.subColumnVector(meanDeviation.getMean());
        DoubleMatrix beforeNormalizationResult = afterMeanApplicationResult.divColumnVector(meanDeviation.getDeviation());
        DoubleMatrix result = beforeNormalizationResult.mulColumnVector(layer.getGammaMatrix()).addiColumnVector(layer.getBetaMatrix());
        ActivationFunctionApplier activationFunctionApplier = layer.getActivationFunctionType().getActivationFunction();
        DoubleMatrix activatedResult = activationFunctionApplier.apply(result);
        BatchNormData newBatchNormData = new BatchNormData(activatedResult, meanDeviationProvider);
        intermediateOutputResult = new IntermediateOutputResult(newBatchNormData, meanDeviation, beforeNormalizationResult, afterMeanApplicationResult);
    }

    @Override
    public void visit(AveragePoolingLayer layer) {

    }

    @Override
    public void visit(MaxPoolingLayer layer) {

    }

    @Override
    public void visit(ConvolutionLayer layer) {

    }

    public IntermediateOutputResult getIntermediateOutputResult() {
        return intermediateOutputResult;
    }
}
