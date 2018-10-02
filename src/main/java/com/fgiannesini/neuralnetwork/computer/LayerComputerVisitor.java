package com.fgiannesini.neuralnetwork.computer;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionApplier;
import com.fgiannesini.neuralnetwork.computer.intermediateoutputcomputer.IntermediateOutputResult;
import com.fgiannesini.neuralnetwork.model.*;
import org.jblas.DoubleMatrix;
import org.jblas.ranges.IntervalRange;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

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
        ConvolutionData data = (ConvolutionData) layerTypeData;
        List<DoubleMatrix> inputs = data.getDatas();
        List<DoubleMatrix> outputs = new ArrayList<>();

        for (DoubleMatrix input : inputs) {
            DoubleMatrix output = computeConvolution(input, DoubleMatrix::mean, layer.getPadding(), layer.getStride(), layer.getFilterSize());
            outputs.add(output);
        }

        intermediateOutputResult = new IntermediateOutputResult(new ConvolutionData(outputs));
    }

    @Override
    public void visit(MaxPoolingLayer layer) {

        ConvolutionData data = (ConvolutionData) layerTypeData;
        List<DoubleMatrix> inputs = data.getDatas();
        List<DoubleMatrix> outputs = new ArrayList<>();

        for (DoubleMatrix input : inputs) {
            DoubleMatrix output = computeConvolution(input, DoubleMatrix::max, layer.getPadding(), layer.getStride(), layer.getFilterSize());
            outputs.add(output);
        }

        intermediateOutputResult = new IntermediateOutputResult(new ConvolutionData(outputs));
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
                    DoubleMatrix convolutedMatrix = computeConvolution(input, inputPart -> inputPart.muli(weights).sum(), layer.getPadding(), layer.getStride(), layer.getFilterSize());
                    if (output == DoubleMatrix.EMPTY) {
                        output = convolutedMatrix;
                    } else {
                        output.addi(convolutedMatrix);
                    }
                }
                output.addi(layer.getBiasMatrices().get(channel));
                outputs.add(output);
            }
        }
        intermediateOutputResult = new IntermediateOutputResult(new ConvolutionData(outputs));
    }

    private DoubleMatrix computeConvolution(DoubleMatrix input, Function<DoubleMatrix, Double> convolutionApplication, int padding, int stride, int filterSize) {
        DoubleMatrix paddedInput = DoubleMatrix.zeros(input.rows + 2 * padding, input.columns + 2 * padding);
        paddedInput.put(new IntervalRange(padding, paddedInput.getRows() - padding), new IntervalRange(padding, paddedInput.getColumns() - padding), input);

        int outputRowCount = (input.getRows() + 2 * padding - filterSize) / stride + 1;
        int outputColumnCount = (input.getColumns() + 2 * padding - filterSize) / stride + 1;
        DoubleMatrix output = DoubleMatrix.zeros(outputRowCount, outputColumnCount);
        for (int rowIndex = 0; rowIndex < paddedInput.getRows() - filterSize + 1; rowIndex += stride) {
            for (int columnIndex = 0; columnIndex < paddedInput.getColumns() - filterSize + 1; columnIndex += stride) {
                DoubleMatrix inputPart = paddedInput.get(new IntervalRange(rowIndex, rowIndex + filterSize), new IntervalRange(columnIndex, columnIndex + filterSize));
                output.put(rowIndex / stride, columnIndex / stride, convolutionApplication.apply(inputPart));
            }
        }

        return output;
    }

    public IntermediateOutputResult getIntermediateOutputResult() {
        return intermediateOutputResult;
    }
}
