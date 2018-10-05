package com.fgiannesini.neuralnetwork.computer.finaloutputcomputer;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionApplier;
import com.fgiannesini.neuralnetwork.computer.data.BatchNormData;
import com.fgiannesini.neuralnetwork.computer.data.LayerTypeData;
import com.fgiannesini.neuralnetwork.computer.data.WeightBiasData;
import com.fgiannesini.neuralnetwork.computer.intermediateoutputcomputer.IntermediateOutputResult;
import com.fgiannesini.neuralnetwork.model.*;
import com.fgiannesini.neuralnetwork.normalizer.meandeviation.MeanDeviation;
import com.fgiannesini.neuralnetwork.normalizer.meandeviation.MeanDeviationProvider;
import org.jblas.DoubleMatrix;

public class LayerComputerWithDropOutRegularizationVisitor implements LayerVisitor {

    private final DoubleMatrix dropOutMatrix;
    private final LayerTypeData layerTypeData;

    private IntermediateOutputResult intermediateOutputResult;

    public LayerComputerWithDropOutRegularizationVisitor(DoubleMatrix dropOutMatrix, LayerTypeData layerTypeData) {
        this.dropOutMatrix = dropOutMatrix;
        this.layerTypeData = layerTypeData;
    }

    public IntermediateOutputResult getIntermediateOutputResult() {
        return intermediateOutputResult;
    }

    @Override
    public void visit(WeightBiasLayer layer) {
        //W.X + b
        WeightBiasData weightBiasData = (WeightBiasData) layerTypeData;
        DoubleMatrix result = layer.getWeightMatrix().mmul(weightBiasData.getData()).addiColumnVector(layer.getBiasMatrix());
        result.muliColumnVector(dropOutMatrix);
        ActivationFunctionApplier activationFunctionApplier = layer.getActivationFunctionType().getActivationFunction();
        DoubleMatrix activatedResult = activationFunctionApplier.apply(result);
        intermediateOutputResult = new IntermediateOutputResult(new WeightBiasData(activatedResult));
    }

    @Override
    public void visit(BatchNormLayer layer) {
        //Z1 = W.X
        BatchNormData batchNormData = (BatchNormData) layerTypeData;
        DoubleMatrix z = layer.getWeightMatrix().mmul(batchNormData.getData());

        MeanDeviationProvider meanDeviationProvider = batchNormData.getMeanDeviationProvider();
        new BatchNormData(z, null).accept(meanDeviationProvider);
        MeanDeviation meanDeviation = meanDeviationProvider.getMeanDeviation();

        //Z2 = (Z1 - mean) / sigma * gamma + beta
        DoubleMatrix afterMeanApplicationResult = z.subColumnVector(meanDeviation.getMean());
        DoubleMatrix beforeNormalizationResult = afterMeanApplicationResult.divColumnVector(meanDeviation.getDeviation());
        DoubleMatrix result = beforeNormalizationResult.mulColumnVector(layer.getGammaMatrix()).addiColumnVector(layer.getBetaMatrix());
        result.muliColumnVector(dropOutMatrix);
        ActivationFunctionApplier activationFunctionApplier = layer.getActivationFunctionType().getActivationFunction();
        DoubleMatrix activatedResult = activationFunctionApplier.apply(result);
        BatchNormData newData = new BatchNormData(activatedResult, meanDeviationProvider);
        intermediateOutputResult = new IntermediateOutputResult(newData, meanDeviation, beforeNormalizationResult, afterMeanApplicationResult);
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
}
