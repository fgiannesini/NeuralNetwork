package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.layerdataprovider;

import com.fgiannesini.neuralnetwork.computer.LayerTypeData;
import com.fgiannesini.neuralnetwork.computer.MeanDeviation;
import com.fgiannesini.neuralnetwork.model.BatchNormLayer;
import org.jblas.DoubleMatrix;

public class GradientBatchNormLayerProvider extends GradientLayerProvider {

    private final BatchNormLayer layer;
    private final MeanDeviation meanDeviation;
    private final DoubleMatrix beforeNormalisationResults;

    public GradientBatchNormLayerProvider(BatchNormLayer layer, LayerTypeData results, LayerTypeData previousResults, DoubleMatrix beforeNormalisationResults, MeanDeviation meanDeviation, int layerIndex) {
        super(results, previousResults, layer, layerIndex, layer.getActivationFunctionType());
        this.layer = layer;
        this.meanDeviation = meanDeviation;
        this.beforeNormalisationResults = beforeNormalisationResults;
    }

    public DoubleMatrix getGammaMatrix() {
        return layer.getGammaMatrix();
    }

    public DoubleMatrix getStandardDeviation() {
        return meanDeviation.getDeviation();
    }

    public DoubleMatrix getBeforeNormalisationCurrentResult() {
        return beforeNormalisationResults;
    }

}
