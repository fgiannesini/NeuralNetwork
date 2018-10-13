package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.layerdataprovider;

import com.fgiannesini.neuralnetwork.computer.intermediateoutputcomputer.IntermediateOutputResult;
import com.fgiannesini.neuralnetwork.model.Layer;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;

import java.util.ArrayList;
import java.util.List;

public class GradientLayerProviderBuilder {

    private NeuralNetworkModel neuralNetworkModel;
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
        checkInputs();
        List<GradientLayerProvider> gradientLayerProviders = new ArrayList<>();

        List<Layer> layers = neuralNetworkModel.getLayers();
        for (int i = 0; i < layers.size(); i++) {
            GradientLayerProviderVisitor layerProviderVisitor = new GradientLayerProviderVisitor(intermediateOutputResultList, i);
            layers.get(i).accept(layerProviderVisitor);
            gradientLayerProviders.add(layerProviderVisitor.getGradientLayerProvider());
        }
        return gradientLayerProviders;
    }

    private void checkInputs() {
        if (intermediateOutputResultList == null) {
            throw new IllegalArgumentException("Intermediate results are not presents");
        }
    }
}
