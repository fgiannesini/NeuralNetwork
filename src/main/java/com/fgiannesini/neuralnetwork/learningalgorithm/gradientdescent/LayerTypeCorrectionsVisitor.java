package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent;

import com.fgiannesini.neuralnetwork.computer.data.*;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.container.GradientDescentCorrection;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.layerdataprovider.GradientBatchNormLayerProvider;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.layerdataprovider.GradientLayerProvider;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.processprovider.BatchNormBackwardReturn;
import com.fgiannesini.neuralnetwork.math.ConvolutionComputer;
import com.fgiannesini.neuralnetwork.model.*;
import org.jblas.DoubleMatrix;
import org.jblas.ranges.IntervalRange;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class LayerTypeCorrectionsVisitor implements DataVisitor {
    private final GradientLayerProvider gradientLayerProvider;
    private LayerTypeData nextGradientLayerProvider;
    private GradientDescentCorrection correction;

    public LayerTypeCorrectionsVisitor(GradientLayerProvider gradientLayerProvider) {
        this.gradientLayerProvider = gradientLayerProvider;
    }

    @Override
    public void visit(WeightBiasData error) {
        DoubleMatrix errorInput = error.getData();
        int inputCount = errorInput.getColumns();
        WeightBiasData previousResult = (WeightBiasData) gradientLayerProvider.getPreviousResult();
        DoubleMatrix weightCorrection = computeWeightCorrection(previousResult.getData(), errorInput, inputCount);
        DoubleMatrix biasCorrection = computeBiasCorrection(errorInput, inputCount);
        correction = new GradientDescentCorrection(weightCorrection, biasCorrection);

        //dZ1 = W2t * dZ2
        WeightBiasLayer layer = (WeightBiasLayer) gradientLayerProvider.getLayer();
        DoubleMatrix nextError = layer.getWeightMatrix().transpose()
                .mmul(error.getData());
        nextGradientLayerProvider = new WeightBiasData(nextError);
    }

    private DoubleMatrix computeBiasCorrection(DoubleMatrix dz, int inputCount) {
        //dB = sum(dZ) ./ m
        return dz.rowSums()
                .divi(inputCount);
    }

    private DoubleMatrix computeWeightCorrection(DoubleMatrix previousLayerResult, DoubleMatrix dz, int inputCount) {
        //dW1 = dZ1 * A0t ./m
        return dz
                .mmul(previousLayerResult.transpose())
                .divi(inputCount);
    }

    @Override
    public void visit(BatchNormData error) {
        DoubleMatrix errorInput = error.getData();
        int inputCount = errorInput.getColumns();
        BatchNormBackwardReturn batchNormBackwardReturn = getBatchNormBackwardReturn(inputCount, (GradientBatchNormLayerProvider) gradientLayerProvider, errorInput);
        correction = batchNormBackwardReturn.getCorrections();

        //dZ1 = W2t * dZ2
        BatchNormLayer layer = (BatchNormLayer) gradientLayerProvider.getLayer();
        DoubleMatrix nextError = layer.getWeightMatrix().transpose()
                .mmul(batchNormBackwardReturn.getNextError());

        nextGradientLayerProvider = new BatchNormData(nextError, error.getMeanDeviationProvider());
    }

    private BatchNormBackwardReturn getBatchNormBackwardReturn(int inputCount, GradientBatchNormLayerProvider gradientLayerProvider, DoubleMatrix dz) {
//        https://kevinzakka.github.io/2016/09/14/batch_normalization/
//        dxhat = dout * gamma
        DoubleMatrix dXhat = dz.mulColumnVector(gradientLayerProvider.getGammaMatrix());
        DoubleMatrix beforeActivationResult = gradientLayerProvider.getBeforeNormalisationCurrentResult();
//        dx = (1. / N) * inv_var * (N * dxhat - np.sum(dxhat, axis = 0)
//                - x_hat * np.sum(dxhat * x_hat, axis = 0))
        DoubleMatrix p1 = dXhat.mul(inputCount);
        DoubleMatrix p2 = dXhat.rowSums();
        DoubleMatrix p3 = beforeActivationResult.mulColumnVector(dXhat.mul(beforeActivationResult).rowSums());
        DoubleMatrix dx = p1.subiColumnVector(p2).subi(p3).divi(inputCount).diviColumnVector(gradientLayerProvider.getStandardDeviation());

//        dbeta = np.sum(dout, axis = 0)
        DoubleMatrix dBeta = dz.rowMeans();
//        dgamma = np.sum(x_hat * dout, axis = 0)
        DoubleMatrix dGamma = dz.mul(beforeActivationResult).rowMeans();

        BatchNormData previousResult = (BatchNormData) gradientLayerProvider.getPreviousResult();
        DoubleMatrix weightCorrection = computeWeightCorrection(previousResult.getData(), dx, inputCount);
        return new BatchNormBackwardReturn(weightCorrection, dGamma, dBeta, dx);
    }

    @Override
    public void visit(ConvolutionData error) {
        ConvolutionData previousResult = (ConvolutionData) gradientLayerProvider.getPreviousResult();
        ConvolutionLayer convolutionLayer = (ConvolutionLayer) gradientLayerProvider.getLayer();
            correction = getConvolutionLayerCorrections(error, previousResult, convolutionLayer);
            nextGradientLayerProvider = getNextConvolutionLayerProvider(error, convolutionLayer);
    }

    @Override
    public void visit(AveragePoolingData error) {
        AveragePoolingLayer averagePoolingLayer = (AveragePoolingLayer) gradientLayerProvider.getLayer();
        correction = new GradientDescentCorrection(IntStream.range(0, averagePoolingLayer.getChannelCount()).mapToObj(i -> DoubleMatrix.EMPTY).collect(Collectors.toList()));
        nextGradientLayerProvider = getNextAveragePoolingLayerProvider(error, averagePoolingLayer);
    }

    private LayerTypeData getNextAveragePoolingLayerProvider(AveragePoolingData input, AveragePoolingLayer averagePoolingLayer) {
        List<DoubleMatrix> inputDatas = input.getDatas();
        List<DoubleMatrix> outputs = inputDatas.stream()
                .map(m -> {
                    int filterAddedColumn = averagePoolingLayer.getFilterSize() - averagePoolingLayer.getFilterSize() % 2;
                    int outputRow = m.getRows() + filterAddedColumn;
                    int outputColumn = m.getColumns() + filterAddedColumn;
                    DoubleMatrix output = DoubleMatrix.zeros(outputRow, outputColumn);
                    output.put(
                            new IntervalRange(averagePoolingLayer.getFilterSize() / 2, outputRow - averagePoolingLayer.getFilterSize() / 2),
                            new IntervalRange(averagePoolingLayer.getFilterSize() / 2, outputColumn - averagePoolingLayer.getFilterSize() / 2),
                            m
                    );
//                    output.divi(averagePoolingLayer.getFilterSize() * averagePoolingLayer.getFilterSize());
                    return applyStride(output, averagePoolingLayer.getStride());
                })
                .collect(Collectors.toList());
        return new ConvolutionData(outputs);
    }

    @Override
    public void visit(MaxPoolingData error) {
        MaxPoolingLayer maxPoolingLayer = (MaxPoolingLayer) gradientLayerProvider.getLayer();
        correction = new GradientDescentCorrection(IntStream.range(0, maxPoolingLayer.getChannelCount()).mapToObj(i -> DoubleMatrix.EMPTY).collect(Collectors.toList()));
        nextGradientLayerProvider = getNextMaxPoolingLayerProvider(error, maxPoolingLayer);
    }
    //Mauvaise implem

    private LayerTypeData getNextMaxPoolingLayerProvider(MaxPoolingData input, MaxPoolingLayer maxPoolingLayer) {
        List<DoubleMatrix> inputDatas = input.getDatas();
        List<DoubleMatrix> outputs = inputDatas.stream()
                .map(m -> {
                    int filterAddedColumn = maxPoolingLayer.getFilterSize() - maxPoolingLayer.getFilterSize() % 2;
                    int outputRow = m.getRows() + filterAddedColumn;
                    int outputColumn = m.getColumns() + filterAddedColumn;
                    DoubleMatrix output = DoubleMatrix.zeros(outputRow, outputColumn);
                    output.put(
                            new IntervalRange(maxPoolingLayer.getFilterSize() / 2, outputRow - maxPoolingLayer.getFilterSize() / 2),
                            new IntervalRange(maxPoolingLayer.getFilterSize() / 2, outputColumn - maxPoolingLayer.getFilterSize() / 2),
                            m
                    );
//                    output.divi(maxPoolingLayer.getFilterSize() * maxPoolingLayer.getFilterSize());
                    return applyStride(output, maxPoolingLayer.getStride());
                })
                .collect(Collectors.toList());
        return new ConvolutionData(outputs);
    }

    private LayerTypeData getNextConvolutionLayerProvider(ConvolutionData error, ConvolutionLayer layer) {

        List<DoubleMatrix> inputs = error.getDatas();
        List<DoubleMatrix> weightMatrices = layer.getWeightMatrices().stream()
                .map(DoubleMatrix::transpose)
                .collect(Collectors.toList());
        int inputCount = inputs.size() / layer.getInputChannelCount();
        List<DoubleMatrix> outputs = new ArrayList<>();

        for (int inputIndex = 0; inputIndex < inputCount; inputIndex++) {
            for (int channel = 0, weightIndex = 0; channel < layer.getOutputChannelCount(); channel++) {
                DoubleMatrix output = DoubleMatrix.EMPTY;
                for (int inputChannelIndex = inputIndex * layer.getInputChannelCount(); inputChannelIndex < (inputIndex + 1) * layer.getInputChannelCount(); inputChannelIndex++, weightIndex++) {
                    DoubleMatrix input = inputs.get(inputChannelIndex);
                    DoubleMatrix weights = weightMatrices.get(weightIndex);
                    DoubleMatrix convolutedMatrix = ConvolutionComputer.get().computeConvolution(input, (inputPart, coord) -> inputPart.muli(weights).sum(), layer.getFilterSize() / 2, 1, layer.getFilterSize());
                    convolutedMatrix = applyStride(convolutedMatrix, layer.getStride());
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

        return new ConvolutionData(outputs);
    }

    private DoubleMatrix applyStride(DoubleMatrix input, int stride) {
        if (stride == 1) {
            return input;
        }
        DoubleMatrix output = DoubleMatrix.zeros(input.getRows() * stride - 1, input.getColumns() * stride - 1);
        for (int i = 0; i < input.getRows(); i++) {
            for (int j = 0; j < input.getColumns(); j++) {
                output.put(i * stride, j * stride, input.get(i, j));
            }
        }
        return output;
    }

    private GradientDescentCorrection getConvolutionLayerCorrections(ConvolutionData error, ConvolutionData previousResult, ConvolutionLayer layer) {
        List<DoubleMatrix> previousResultDatas = previousResult.getDatas();
        List<DoubleMatrix> errorDatas = error.getDatas();
        int inputCount = errorDatas.size() / layer.getOutputChannelCount();
        List<DoubleMatrix> weightCorrections = IntStream.range(0, layer.getInputChannelCount() * layer.getOutputChannelCount())
                .mapToObj(i -> DoubleMatrix.zeros(layer.getFilterSize(), layer.getFilterSize()))
                .collect(Collectors.toList());
        List<DoubleMatrix> biasCorrections = IntStream.range(0, layer.getOutputChannelCount())
                .mapToObj(i -> DoubleMatrix.zeros(1, 1))
                .collect(Collectors.toList());
        for (int inputIndex = 0; inputIndex < inputCount; inputIndex++) {
            for (int i = 0; i < layer.getInputChannelCount(); i++) {
                int previousResultIndex = inputIndex * layer.getInputChannelCount() + i;
                for (int j = 0; j < layer.getOutputChannelCount(); j++) {
                    int errorIndex = inputIndex * layer.getOutputChannelCount() + j;
                    DoubleMatrix errorData = errorDatas.get(errorIndex);
                    DoubleMatrix stridedErrorData = applyStride(errorData, layer.getStride());
                    DoubleMatrix weightCorrection = ConvolutionComputer.get().computeConvolution(previousResultDatas.get(previousResultIndex), (inputPart, coord) -> inputPart.muli(stridedErrorData).sum(), layer.getPadding(), 1, stridedErrorData.getRows());
                    weightCorrections.get(i + j * layer.getInputChannelCount()).addi(weightCorrection);
                    biasCorrections.get(j).addi(errorData.sum());
                }
            }
        }
        weightCorrections.forEach(m -> m.divi(inputCount));
        biasCorrections.forEach(m -> m.divi(inputCount * layer.getInputChannelCount()));

        GradientDescentCorrection gradientDescentCorrection = new GradientDescentCorrection(weightCorrections);
        gradientDescentCorrection.addCorrectionResults(biasCorrections);
        return gradientDescentCorrection;
    }

    public GradientDescentCorrection getCorrection() {
        return correction;
    }

    public LayerTypeData getNextGradientLayerProvider() {
        return nextGradientLayerProvider;
    }
}
