package com.fgiannesini.neuralnetwork.computer;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionApplier;
import com.fgiannesini.neuralnetwork.computer.intermediateoutputcomputer.IntermediateOutputResult;
import com.fgiannesini.neuralnetwork.model.*;
import org.jblas.DoubleMatrix;
import org.jblas.ranges.IntervalRange;

import java.util.ArrayList;
import java.util.List;

public class LayerComputerVisitor implements LayerVisitor {

    private final LayerTypeData layerTypeData;
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
        ConvolutionData data = (ConvolutionData) layerTypeData;
        List<DoubleMatrix> inputs = data.getDatas();
        List<DoubleMatrix> weightMatrices = layer.getWeightMatrices();
        List<DoubleMatrix> outputs = new ArrayList<>();
        for (int channel = 0; channel < layer.getChannelCount(); channel++) {
            for (int inputIndex = 0; inputIndex < inputs.size(); inputIndex++) {
                DoubleMatrix input = inputs.get(inputIndex);
                DoubleMatrix weights = weightMatrices.get(inputs.size() * channel + inputIndex);
                DoubleMatrix output = computeConvolution(input, weights, layer.getPadding(), layer.getStride());
                outputs.add(output);
            }
        }
        intermediateOutputResult = new IntermediateOutputResult(new ConvolutionData(outputs));
    }

    private DoubleMatrix computeConvolution(DoubleMatrix input, DoubleMatrix weights, int padding, int stride) {
        DoubleMatrix paddedInput = DoubleMatrix.zeros(input.rows + 2 * padding, input.columns + 2 * padding);
        paddedInput.put(new IntervalRange(padding, paddedInput.getRows() - padding), new IntervalRange(padding, paddedInput.getColumns() - padding), input);

        int outputRowCount = (input.getRows() + 2 * padding - weights.getRows()) / stride + 1;
        int outputColumnCount = (input.getColumns() + 2 * padding - weights.getColumns()) / stride + 1;
        DoubleMatrix output = DoubleMatrix.zeros(outputRowCount, outputColumnCount);
        for (int rowIndex = 0; rowIndex < paddedInput.getRows() - weights.getRows() + 1; rowIndex += stride) {
            for (int columnIndex = 0; columnIndex < paddedInput.getColumns() - weights.getColumns() + 1; columnIndex += stride) {
                DoubleMatrix inputPart = paddedInput.get(new IntervalRange(rowIndex, rowIndex + weights.getRows()), new IntervalRange(columnIndex, columnIndex + weights.getColumns()));
                output.put(rowIndex, columnIndex, inputPart.muli(weights).sum());
            }
        }

        return output;
    }

    public IntermediateOutputResult getIntermediateOutputResult() {
        return intermediateOutputResult;
    }
}
