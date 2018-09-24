package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.layerdataprovider;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionApplier;
import com.fgiannesini.neuralnetwork.computer.MeanDeviation;
import com.fgiannesini.neuralnetwork.model.BatchNormLayer;
import com.fgiannesini.neuralnetwork.model.Layer;
import org.jblas.DoubleMatrix;

import java.util.List;
import java.util.stream.Collectors;

public class GradientBatchNormLayerProvider extends GradientLayerProvider {

    private List<BatchNormLayer> layers;
    private List<MeanDeviation> meanDeviations;
    private List<DoubleMatrix> beforeNormalisationResults;

    private List<DoubleMatrix> afterMeanApplicationResult;

    public GradientBatchNormLayerProvider(List<Layer> layers, List<DoubleMatrix> results, List<DoubleMatrix> beforeNormalisationResults, List<MeanDeviation> meanDeviations, List<DoubleMatrix> afterMeanApplicationResult) {
        super(results, layers.size());
        this.layers = layers.stream().map(BatchNormLayer.class::cast).collect(Collectors.toList());
        this.meanDeviations = meanDeviations;
        this.beforeNormalisationResults = beforeNormalisationResults;
        this.afterMeanApplicationResult = afterMeanApplicationResult;
    }

    public DoubleMatrix getGammaMatrix() {
        return layers.get(currentLayerIndex - 1).getGammaMatrix();
    }

    public DoubleMatrix getStandardDeviation() {
        return meanDeviations.get(currentLayerIndex).getDeviation();
    }

    public int getInputSize() {
        return layers.get(currentLayerIndex - 1).getInputLayerSize();
    }

    public int getOutputSize() {
        return layers.get(currentLayerIndex - 1).getOutputLayerSize();
    }

    public DoubleMatrix getBeforeNormalisationCurrentResult() {
        return beforeNormalisationResults.get(currentLayerIndex);
    }

    public DoubleMatrix getAfterMeanApplicationCurrentResult() {
        return afterMeanApplicationResult.get(currentLayerIndex);
    }

    public DoubleMatrix getPreviousWeightMatrix() {
        return layers.get(currentLayerIndex).getWeightMatrix();
    }

    public ActivationFunctionApplier getCurrentActivationFunction() {
        return layers.get(currentLayerIndex - 1).getActivationFunctionType().getActivationFunction();
    }
}
