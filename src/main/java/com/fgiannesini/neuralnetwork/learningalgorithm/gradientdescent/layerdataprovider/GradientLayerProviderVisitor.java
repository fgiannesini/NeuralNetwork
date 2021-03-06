package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.layerdataprovider;

import com.fgiannesini.neuralnetwork.computer.data.LayerTypeData;
import com.fgiannesini.neuralnetwork.computer.data.WeightBiasData;
import com.fgiannesini.neuralnetwork.computer.data.adapter.ForwardDataAdapterVisitor;
import com.fgiannesini.neuralnetwork.computer.intermediateoutputcomputer.IntermediateOutputResult;
import com.fgiannesini.neuralnetwork.model.*;

import java.util.List;

public class GradientLayerProviderVisitor implements LayerVisitor {

    private final List<IntermediateOutputResult> intermediateOutputResult;
    private final int layerIndex;
    private GradientLayerProvider gradientLayerProvider;

    public GradientLayerProviderVisitor(List<IntermediateOutputResult> intermediateOutputResult, int layerIndex) {
        this.intermediateOutputResult = intermediateOutputResult;
        this.layerIndex = layerIndex;
    }

    @Override
    public void visit(WeightBiasLayer layer) {
        WeightBiasData result = (WeightBiasData) this.intermediateOutputResult.get(layerIndex + 1).getResult();
        WeightBiasData previousResult = (WeightBiasData) formatData(layer, this.intermediateOutputResult.get(layerIndex));
        gradientLayerProvider = new GradientWeightBiasLayerProvider(layer, result, previousResult, layerIndex);
    }

    @Override
    public void visit(BatchNormLayer layer) {
        IntermediateOutputResult intermediateOutputResult = this.intermediateOutputResult.get(layerIndex + 1);
        LayerTypeData previousData = formatData(layer, this.intermediateOutputResult.get(layerIndex));
        gradientLayerProvider = new GradientBatchNormLayerProvider(layer, intermediateOutputResult.getResult(), previousData, intermediateOutputResult.getBeforeNormalisationResult(), intermediateOutputResult.getMeanDeviation(), layerIndex);
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
        LayerTypeData previousData = formatData(layer, this.intermediateOutputResult.get(layerIndex));
        return new GradientConvolutionLayerProvider(layer, intermediateOutputResult.getResult(), previousData, layerIndex);
    }

    private LayerTypeData formatData(Layer layer, IntermediateOutputResult intermediateOutputResult) {
        ForwardDataAdapterVisitor adapterVisitor = new ForwardDataAdapterVisitor(intermediateOutputResult.getResult());
        layer.accept(adapterVisitor);
        return adapterVisitor.getData();
    }

    public GradientLayerProvider getGradientLayerProvider() {
        return gradientLayerProvider;
    }
}
