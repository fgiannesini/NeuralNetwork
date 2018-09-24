package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.layerdataprovider;

import com.fgiannesini.neuralnetwork.computer.BatchNormData;
import com.fgiannesini.neuralnetwork.computer.MeanDeviation;
import com.fgiannesini.neuralnetwork.computer.WeightBiasData;
import com.fgiannesini.neuralnetwork.computer.intermediateoutputcomputer.IntermediateOutputResult;
import com.fgiannesini.neuralnetwork.model.BatchNormLayer;
import com.fgiannesini.neuralnetwork.model.Layer;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import com.fgiannesini.neuralnetwork.model.WeightBiasLayer;
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

    public GradientLayerProvider build() {
        checkData();

        GradientLayerProvider gradientLayerProvider;
        Layer firstLayer = neuralNetworkModel.getLayers().get(0);
        if (firstLayer instanceof WeightBiasLayer) {
            if (intermediateOutputResultList != null) {
                List<DoubleMatrix> weightBiasResults = intermediateOutputResultList.stream()
                        .map(intermediateOutputResult -> ((WeightBiasData) intermediateOutputResult.getResult()).getInput())
                        .collect(Collectors.toList());
                gradientLayerProvider = new GradientWeightBiasLayerProvider(neuralNetworkModel.getLayers(), weightBiasResults);
            } else {
                gradientLayerProvider = new GradientWeightBiasLayerProvider(neuralNetworkModel.getLayers(), this.results);
            }
        } else if (firstLayer instanceof BatchNormLayer) {
            List<DoubleMatrix> batchNormResults = intermediateOutputResultList.stream()
                    .map(intermediateOutputResult -> ((BatchNormData) intermediateOutputResult.getResult()).getInput())
                    .collect(Collectors.toList());
            List<MeanDeviation> meanDeviations = intermediateOutputResultList.stream().map(IntermediateOutputResult::getMeanDeviation).collect(Collectors.toList());
            List<DoubleMatrix> afterMeanApplicationResults = intermediateOutputResultList.stream().map(IntermediateOutputResult::getAfterMeanApplicationResult).collect(Collectors.toList());
            List<DoubleMatrix> beforeNormalizationResults = intermediateOutputResultList.stream().map(IntermediateOutputResult::getBeforeNormalisationResult).collect(Collectors.toList());
            gradientLayerProvider = new GradientBatchNormLayerProvider(neuralNetworkModel.getLayers(), batchNormResults, beforeNormalizationResults, meanDeviations, afterMeanApplicationResults);
        } else {
            throw new IllegalArgumentException(firstLayer.getClass().getSimpleName() + " layerType is not managed");
        }
        return gradientLayerProvider;
    }

    private void checkData() {
        if (neuralNetworkModel == null) {
            throw new IllegalArgumentException("Neural network model should be set");
        }
        Layer firstLayer = neuralNetworkModel.getLayers().get(0);
        if (firstLayer instanceof BatchNormLayer && intermediateOutputResultList == null) {
            throw new IllegalArgumentException(BatchNormLayer.class.getSimpleName() + " should be associated to intermediateOutputResults data");
        }
        if (firstLayer instanceof WeightBiasLayer && results == null && intermediateOutputResultList == null) {
            throw new IllegalArgumentException(WeightBiasLayer.class.getSimpleName() + " should be associated to results or intermediateOutputResults data");
        }
    }
}
