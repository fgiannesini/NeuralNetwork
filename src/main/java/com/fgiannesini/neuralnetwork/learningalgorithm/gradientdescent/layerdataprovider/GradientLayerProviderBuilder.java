package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.layerdataprovider;

import com.fgiannesini.neuralnetwork.computer.intermediateoutputcomputer.IntermediateOutputResult;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import org.jblas.DoubleMatrix;

import java.util.ArrayList;
import java.util.List;

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

    public List<GradientLayerProvider> build() {
        List<GradientLayerProvider> gradientLayerProviders = new ArrayList<>();

        for (int i = 0; i < neuralNetworkModel.getLayers().size(); i++) {
            GradientLayerProviderVisitor layerProviderVisitor = new GradientLayerProviderVisitor(intermediateOutputResultList, 0);
            neuralNetworkModel.getLayers().get(i).accept(layerProviderVisitor);
            gradientLayerProviders.add(layerProviderVisitor.getGradientLayerProvider());
        }
        return gradientLayerProviders;
    }
}
