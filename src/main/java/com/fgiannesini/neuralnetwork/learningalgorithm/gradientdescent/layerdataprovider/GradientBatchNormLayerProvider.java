package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.layerdataprovider;

import com.fgiannesini.neuralnetwork.computer.MeanDeviation;
import com.fgiannesini.neuralnetwork.model.BatchNormLayer;
import org.jblas.DoubleMatrix;

import java.util.List;

public class GradientBatchNormLayerProvider extends GradientLayerProvider<BatchNormLayer> {

    private List<MeanDeviation> meanDeviations;

    public GradientBatchNormLayerProvider(List<BatchNormLayer> layers, List<DoubleMatrix> results, List<MeanDeviation> meanDeviations) {
        super(layers, results);
        this.meanDeviations = meanDeviations;
    }

    public DoubleMatrix getGammaMatrix() {
        return layers.get(currentLayerIndex - 1).getGammaMatrix();
    }

    public DoubleMatrix getStandardDeviation() {
        return meanDeviations.get(currentLayerIndex).getDeviation();
    }

    public DoubleMatrix getMean() {
        return meanDeviations.get(currentLayerIndex).getMean();
    }

    public int getInputSize() {
        return layers.get(currentLayerIndex - 1).getInputLayerSize();
    }

    public int getOutputSize() {
        return layers.get(currentLayerIndex - 1).getOutputLayerSize();
    }
}
