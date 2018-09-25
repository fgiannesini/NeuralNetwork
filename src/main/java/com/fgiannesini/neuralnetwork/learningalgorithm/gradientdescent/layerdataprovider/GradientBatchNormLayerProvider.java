package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.layerdataprovider;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionApplier;
import com.fgiannesini.neuralnetwork.computer.MeanDeviation;
import com.fgiannesini.neuralnetwork.model.BatchNormLayer;
import org.jblas.DoubleMatrix;

public class GradientBatchNormLayerProvider extends GradientLayerProvider {

    private BatchNormLayer layer;
    private MeanDeviation meanDeviation;
    private DoubleMatrix beforeNormalisationResults;

    public GradientBatchNormLayerProvider(BatchNormLayer layer, DoubleMatrix results, DoubleMatrix previousResults, DoubleMatrix beforeNormalisationResults, MeanDeviation meanDeviation) {
        super(results, previousResults, layer.getActivationFunctionType());
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

    public DoubleMatrix getPreviousWeightMatrix() {
        return layer.getWeightMatrix();
    }

    public ActivationFunctionApplier getCurrentActivationFunction() {
        return layer.getActivationFunctionType().getActivationFunction();
    }
}
