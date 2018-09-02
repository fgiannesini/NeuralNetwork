package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent;

import com.fgiannesini.neuralnetwork.model.BatchNormLayer;
import org.jblas.DoubleMatrix;

import java.util.ArrayList;
import java.util.List;

public class GradientBatchNormLayerProvider extends GradientLayerProvider<BatchNormLayer> {

    private List<DoubleMatrix> means;
    private List<DoubleMatrix> standardDeviations;

    public GradientBatchNormLayerProvider(List<BatchNormLayer> layers, List<DoubleMatrix> results) {
        super(layers, results);
        means = new ArrayList<>();
        standardDeviations = new ArrayList<>();
    }

    public DoubleMatrix getGammaMatrix() {
        return layers.get(currentLayerIndex - 1).getGammaMatrix();
    }

    public void addMean(DoubleMatrix mean) {
        means.add(mean);
    }

    public void addStandardDeviation(DoubleMatrix standardDeviation) {
        standardDeviations.add(standardDeviation);
    }

    public DoubleMatrix getStandardDeviation() {
        return standardDeviations.get(currentLayerIndex);
    }

    public DoubleMatrix getMean() {
        return means.get(currentLayerIndex);
    }
}
