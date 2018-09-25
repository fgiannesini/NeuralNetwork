package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.layerdataprovider;

import com.fgiannesini.neuralnetwork.computer.BatchNormData;
import com.fgiannesini.neuralnetwork.computer.WeightBiasData;
import com.fgiannesini.neuralnetwork.computer.intermediateoutputcomputer.IntermediateOutputResult;
import com.fgiannesini.neuralnetwork.model.*;
import org.jblas.DoubleMatrix;

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
        DoubleMatrix result = ((WeightBiasData) intermediateOutputResult.get(layerIndex).getResult()).getInput();
        DoubleMatrix previousResult = null;
        if (layerIndex > 0) {
            previousResult = ((WeightBiasData) intermediateOutputResult.get(layerIndex - 1).getResult()).getInput();
        }
        gradientLayerProvider = new GradientWeightBiasLayerProvider(layer, result, previousResult);
    }

    @Override
    public void visit(BatchNormLayer layer) {
        IntermediateOutputResult intermediateOutputResult = this.intermediateOutputResult.get(layerIndex);
        DoubleMatrix result = ((BatchNormData) intermediateOutputResult.getResult()).getInput();
        DoubleMatrix previousResult = null;
        if (layerIndex > 0) {
            previousResult = ((BatchNormData) this.intermediateOutputResult.get(layerIndex - 1).getResult()).getInput();
        }
        gradientLayerProvider = new GradientBatchNormLayerProvider(layer, result, previousResult, intermediateOutputResult.getBeforeNormalisationResult(), intermediateOutputResult.getMeanDeviation());
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
