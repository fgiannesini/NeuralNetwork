package com.fgiannesini.neuralnetwork.learningalgorithm.regularization.l2;

import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.container.*;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.layerdataprovider.GradientLayerProvider;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.processprovider.IGradientDescentProcessProvider;
import com.fgiannesini.neuralnetwork.model.Layer;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;

import java.util.List;
import java.util.function.Function;

public class GradientDescentWithL2RegularizationProcessProvider<L extends Layer> implements IGradientDescentProcessProvider<L> {

    private final IGradientDescentProcessProvider<L> gradientDescentProcessProvider;
    private final double regularizationCoeff;
    private final NeuralNetworkModel<L> originalNeuralNetworkModel;

    public GradientDescentWithL2RegularizationProcessProvider(double regularizationCoeff, NeuralNetworkModel<L> originalNeuralNetworkModel, IGradientDescentProcessProvider<L> gradientDescentProcessProvider) {
        this.regularizationCoeff = regularizationCoeff;
        this.originalNeuralNetworkModel = originalNeuralNetworkModel.clone();
        this.gradientDescentProcessProvider = gradientDescentProcessProvider;
    }

    @Override
    public Function<GradientDescentCorrectionsContainer<L>, GradientDescentCorrectionsContainer<L>> getGradientDescentCorrectionsLauncher() {
        return gradientDescentProcessProvider.getGradientDescentCorrectionsLauncher()
                .andThen(gradientDescentCorrectionsContainer -> {
                    NeuralNetworkModel<L> neuralNetworkModel = gradientDescentCorrectionsContainer.getCorrectedNeuralNetworkModel();
                    List<L> layers = neuralNetworkModel.getLayers();
                    List<L> originalLayers = originalNeuralNetworkModel.getLayers();
                    for (int layerIndex = 0; layerIndex < layers.size(); layerIndex++) {
                        L layer = layers.get(layerIndex);
                        L originalLayer = originalLayers.get(layerIndex);
                        layer.getWeightMatrix().subi(originalLayer.getWeightMatrix().mul(gradientDescentCorrectionsContainer.getLearningRate() * regularizationCoeff / gradientDescentCorrectionsContainer.getInputCount()));
                    }
                    return new GradientDescentCorrectionsContainer<>(neuralNetworkModel, gradientDescentCorrectionsContainer.getGradientDescentCorrections(), gradientDescentCorrectionsContainer.getInputCount(), gradientDescentCorrectionsContainer.getLearningRate());
                });
    }

    @Override
    public Function<BackwardComputationContainer, List<GradientDescentCorrection>> getBackwardComputationLauncher() {
        return gradientDescentProcessProvider.getBackwardComputationLauncher();
    }

    @Override
    public Function<ErrorComputationContainer, ErrorComputationContainer> getErrorComputationLauncher() {
        return gradientDescentProcessProvider.getErrorComputationLauncher();
    }

    @Override
    public Function<ErrorComputationContainer, ErrorComputationContainer> getFirstErrorComputationLauncher() {
        return gradientDescentProcessProvider.getFirstErrorComputationLauncher();
    }

    @Override
    public Function<ForwardComputationContainer<L>, GradientLayerProvider<L>> getForwardComputationLauncher() {
        return gradientDescentProcessProvider.getForwardComputationLauncher();
    }

    @Override
    public Function<DataContainer, DataContainer> getDataProcessLauncher() {
        return Function.identity();
    }
}
