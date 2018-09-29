package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.layerdataprovider;

import com.fgiannesini.neuralnetwork.computer.BatchNormData;
import com.fgiannesini.neuralnetwork.computer.WeightBiasData;
import com.fgiannesini.neuralnetwork.computer.intermediateoutputcomputer.IntermediateOutputResult;
import com.fgiannesini.neuralnetwork.model.*;
import org.jblas.DoubleMatrix;

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
        DoubleMatrix result = ((WeightBiasData) intermediateOutputResult.get(layerIndex + 1).getResult()).getInput();
        DoubleMatrix previousResult = ((WeightBiasData) intermediateOutputResult.get(layerIndex).getResult()).getInput();
        WeightBiasLayer previousLayer = null;
        if (layerIndex + 1 < layers.size()) {
            previousLayer = (WeightBiasLayer) layers.get(layerIndex + 1);
        }
        gradientLayerProvider = new GradientWeightBiasLayerProvider(layer, previousLayer, result, previousResult, layerIndex);
    }

    @Override
    public void visit(BatchNormLayer layer) {
        IntermediateOutputResult intermediateOutputResult = this.intermediateOutputResult.get(layerIndex + 1);
        DoubleMatrix result = ((BatchNormData) intermediateOutputResult.getResult()).getInput();
        DoubleMatrix previousResult = ((BatchNormData) this.intermediateOutputResult.get(layerIndex).getResult()).getInput();
        BatchNormLayer previousLayer = null;
        if (layerIndex + 1 < layers.size()) {
            previousLayer = (BatchNormLayer) layers.get(layerIndex + 1);
        }
        gradientLayerProvider = new GradientBatchNormLayerProvider(layer, previousLayer, result, previousResult, intermediateOutputResult.getBeforeNormalisationResult(), intermediateOutputResult.getMeanDeviation(), layerIndex);

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

    public GradientLayerProvider getGradientLayerProvider() {
        return gradientLayerProvider;
    }
}
