package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.layerdataprovider;

import com.fgiannesini.neuralnetwork.computer.MeanDeviation;
import com.fgiannesini.neuralnetwork.computer.intermediateoutputcomputer.IntermediateOutputResult;
import com.fgiannesini.neuralnetwork.model.Layer;
import com.fgiannesini.neuralnetwork.model.LayerType;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import org.jblas.DoubleMatrix;

import java.util.List;
import java.util.stream.Collectors;

public class GradientLayerProviderBuilder {

    private NeuralNetworkModel neuralNetworkModel;
    private List<DoubleMatrix> results;
    private List<IntermediateOutputResult> intermediateOutputResultList;

    public static GradientLayerProviderBuilder init() {
        return new GradientLayerProviderBuilder();
    }

    public GradientLayerProviderBuilder withModel(NeuralNetworkModel neuralNetworkModel) {
        this.neuralNetworkModel = neuralNetworkModel;
        return this;
    }

    public GradientLayerProviderBuilder withIntermediateResults(List<IntermediateOutputResult> intermediateOutputResultList) {
        this.intermediateOutputResultList = intermediateOutputResultList;
        return this;
    }

    public GradientLayerProviderBuilder withResults(List<DoubleMatrix> results) {
        this.results = results;
        return this;
    }

    public <L extends Layer> GradientLayerProvider<L> build() {
        checkData();
        LayerType layerType = neuralNetworkModel.getLayerType();
        GradientLayerProvider<L> gradientLayerProvider;
        switch (layerType) {
            case BATCH_NORM:
                List<DoubleMatrix> batchNormResults = intermediateOutputResultList.stream().map(IntermediateOutputResult::getResult).collect(Collectors.toList());
                List<MeanDeviation> meanDeviations = intermediateOutputResultList.stream().map(IntermediateOutputResult::getMeanDeviation).collect(Collectors.toList());
                List<DoubleMatrix> afterMeanApplicationResults = intermediateOutputResultList.stream().map(IntermediateOutputResult::getAfterMeanApplicationResult).collect(Collectors.toList());
                List<DoubleMatrix> beforeNormalizationResults = intermediateOutputResultList.stream().map(IntermediateOutputResult::getBeforeNormalisationResult).collect(Collectors.toList());
                gradientLayerProvider = (GradientLayerProvider<L>) new GradientBatchNormLayerProvider(neuralNetworkModel.getLayers(), batchNormResults, beforeNormalizationResults, meanDeviations, afterMeanApplicationResults);
                break;
            case WEIGHT_BIAS:
                if (intermediateOutputResultList != null) {
                    List<DoubleMatrix> weightBiasResults = intermediateOutputResultList.stream().map(IntermediateOutputResult::getResult).collect(Collectors.toList());
                    gradientLayerProvider = (GradientLayerProvider<L>) new GradientWeightBiasLayerProvider(neuralNetworkModel.getLayers(), weightBiasResults);
                } else {
                    gradientLayerProvider = (GradientLayerProvider<L>) new GradientWeightBiasLayerProvider(neuralNetworkModel.getLayers(), this.results);
                }
                break;
            default:
                throw new IllegalArgumentException(layerType + " layerType is not managed");
        }
        return gradientLayerProvider;
    }

    private void checkData() {
        if (neuralNetworkModel == null) {
            throw new IllegalArgumentException("Neural network model should be set");
        }
        if (LayerType.BATCH_NORM.equals(neuralNetworkModel.getLayerType()) && intermediateOutputResultList == null) {
            throw new IllegalArgumentException(LayerType.BATCH_NORM + " should be associated to intermediateOutputResults data");
        }
        if (LayerType.WEIGHT_BIAS.equals(neuralNetworkModel.getLayerType()) && results == null && intermediateOutputResultList == null) {
            throw new IllegalArgumentException(LayerType.WEIGHT_BIAS + " should be associated to results or intermediateOutputResults data");
        }
    }
}
