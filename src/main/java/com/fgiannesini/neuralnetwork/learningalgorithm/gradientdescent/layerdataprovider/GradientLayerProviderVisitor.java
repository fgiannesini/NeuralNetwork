package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.layerdataprovider;

import com.fgiannesini.neuralnetwork.computer.LayerTypeData;
import com.fgiannesini.neuralnetwork.computer.WeightBiasData;
import com.fgiannesini.neuralnetwork.computer.intermediateoutputcomputer.DataAdapterVisitor;
import com.fgiannesini.neuralnetwork.computer.intermediateoutputcomputer.IntermediateOutputResult;
import com.fgiannesini.neuralnetwork.model.*;

import java.util.List;

public class GradientLayerProviderVisitor implements LayerVisitor {

    private final List<Layer> layers;
    private final List<IntermediateOutputResult> intermediateOutputResult;
    private final int layerIndex;
    private GradientLayerProvider gradientLayerProvider;

    public GradientLayerProviderVisitor(List<Layer> layers, List<IntermediateOutputResult> intermediateOutputResult, int layerIndex) {
        this.layers = layers;
        this.intermediateOutputResult = intermediateOutputResult;
        this.layerIndex = layerIndex;
    }

    @Override
    public void visit(WeightBiasLayer layer) {
        WeightBiasData result = (WeightBiasData) this.intermediateOutputResult.get(layerIndex + 1).getResult();
        WeightBiasData previousResult = (WeightBiasData) formatData(layer, this.intermediateOutputResult.get(layerIndex));

        WeightBiasLayer previousLayer = null;
        if (layerIndex + 1 < layers.size()) {
            previousLayer = (WeightBiasLayer) layers.get(layerIndex + 1);
        }
        gradientLayerProvider = new GradientWeightBiasLayerProvider(layer, previousLayer, result, previousResult, layerIndex);
    }

    @Override
    public void visit(BatchNormLayer layer) {
        IntermediateOutputResult intermediateOutputResult = this.intermediateOutputResult.get(layerIndex + 1);
        LayerTypeData previousData = formatData(layer, this.intermediateOutputResult.get(layerIndex));

        BatchNormLayer previousLayer = null;
        if (layerIndex + 1 < layers.size()) {
            previousLayer = (BatchNormLayer) layers.get(layerIndex + 1);
        }
        gradientLayerProvider = new GradientBatchNormLayerProvider(layer, previousLayer, intermediateOutputResult.getResult(), previousData, intermediateOutputResult.getBeforeNormalisationResult(), intermediateOutputResult.getMeanDeviation(), layerIndex);
    }

    @Override
    public void visit(AveragePoolingLayer layer) {
        this.gradientLayerProvider = createGradientConvolutionLayerProvider(layer);
    }

    @Override
    public void visit(MaxPoolingLayer layer) {
        this.gradientLayerProvider = createGradientConvolutionLayerProvider(layer);
    }

    @Override
    public void visit(ConvolutionLayer layer) {
        this.gradientLayerProvider = createGradientConvolutionLayerProvider(layer);
    }

    private GradientConvolutionLayerProvider createGradientConvolutionLayerProvider(Layer layer) {
        IntermediateOutputResult intermediateOutputResult = this.intermediateOutputResult.get(layerIndex + 1);
        IntermediateOutputResult previousIntermediateOutputResult = this.intermediateOutputResult.get(layerIndex);
        return new GradientConvolutionLayerProvider(layer, intermediateOutputResult.getResult(), previousIntermediateOutputResult.getResult(), layerIndex);
    }

    private LayerTypeData formatData(Layer layer, IntermediateOutputResult intermediateOutputResult) {
        DataAdapterVisitor adapterVisitor = new DataAdapterVisitor(intermediateOutputResult.getResult());
        layer.accept(adapterVisitor);
        return adapterVisitor.getData();
    }

    public GradientLayerProvider getGradientLayerProvider() {
        return gradientLayerProvider;
    }
}
