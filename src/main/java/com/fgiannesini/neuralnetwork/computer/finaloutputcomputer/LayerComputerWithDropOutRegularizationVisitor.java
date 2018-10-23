package com.fgiannesini.neuralnetwork.computer.finaloutputcomputer;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionApplier;
import com.fgiannesini.neuralnetwork.computer.LayerComputerVisitor;
import com.fgiannesini.neuralnetwork.computer.data.BatchNormData;
import com.fgiannesini.neuralnetwork.computer.data.ConvolutionData;
import com.fgiannesini.neuralnetwork.computer.data.LayerTypeData;
import com.fgiannesini.neuralnetwork.computer.data.WeightBiasData;
import com.fgiannesini.neuralnetwork.computer.intermediateoutputcomputer.IntermediateOutputResult;
import com.fgiannesini.neuralnetwork.math.ConvolutionComputer;
import com.fgiannesini.neuralnetwork.model.*;
import com.fgiannesini.neuralnetwork.normalizer.meandeviation.BatchNormMeanDeviation;
import com.fgiannesini.neuralnetwork.normalizer.meandeviation.MeanDeviationProvider;
import org.jblas.DoubleMatrix;

import java.util.ArrayList;
import java.util.List;

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
        BatchNormMeanDeviation meanDeviation = (BatchNormMeanDeviation) meanDeviationProvider.getMeanDeviation();

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
        LayerComputerVisitor layerComputerVisitor = new LayerComputerVisitor(layerTypeData);
        layer.accept(layerComputerVisitor);
        intermediateOutputResult = layerComputerVisitor.getIntermediateOutputResult();
    }

    @Override
    public void visit(MaxPoolingLayer layer) {
        LayerComputerVisitor layerComputerVisitor = new LayerComputerVisitor(layerTypeData);
        layer.accept(layerComputerVisitor);
        intermediateOutputResult = layerComputerVisitor.getIntermediateOutputResult();
    }

    @Override
    public void visit(ConvolutionLayer layer) {
        ConvolutionData data = (ConvolutionData) layerTypeData;
        List<DoubleMatrix> inputs = data.getDatas();
        List<DoubleMatrix> weightMatrices = layer.getWeightMatrices();
        int inputCount = inputs.size() / layer.getInputChannelCount();
        List<DoubleMatrix> outputs = new ArrayList<>();

        for (int inputIndex = 0; inputIndex < inputCount; inputIndex++) {
            for (int channel = 0, weightIndex = 0; channel < layer.getOutputChannelCount(); channel++) {
                DoubleMatrix output = DoubleMatrix.EMPTY;
                for (int inputChannelIndex = inputIndex * layer.getInputChannelCount(); inputChannelIndex < (inputIndex + 1) * layer.getInputChannelCount(); inputChannelIndex++, weightIndex++) {
                    DoubleMatrix input = inputs.get(inputChannelIndex);
                    DoubleMatrix weights = weightMatrices.get(weightIndex);
                    DoubleMatrix convolutedMatrix = ConvolutionComputer.get().computeConvolution(input, (inputPart, coord) -> inputPart.muli(weights).sum(), layer.getPadding(), layer.getStride(), layer.getFilterSize());
                    if (output == DoubleMatrix.EMPTY) {
                        output = convolutedMatrix;
                    } else {
                        output.addi(convolutedMatrix);
                    }
                }
                output.addi(layer.getBiasMatrices().get(channel));
                DoubleMatrix result = output.mul(dropOutMatrix);
                ActivationFunctionApplier activationFunctionApplier = layer.getActivationFunctionType().getActivationFunction();
                DoubleMatrix activatedResult = activationFunctionApplier.apply(result);
                outputs.add(activatedResult);
            }
        }
        intermediateOutputResult = new IntermediateOutputResult(new ConvolutionData(outputs, layer.getOutputChannelCount()));
    }
}
