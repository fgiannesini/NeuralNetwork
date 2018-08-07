package com.fgiannesini.neuralnetwork.learningalgorithm.regularization.l2;

import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.*;
import com.fgiannesini.neuralnetwork.model.Layer;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;

import java.util.List;
import java.util.function.Function;

public class GradientDescentWithL2RegularizationProcessProvider implements IGradientDescentProcessProvider {

    private final IGradientDescentProcessProvider gradientDescentProcessProvider;
    private final double regularizationCoeff;
    private final NeuralNetworkModel originalNeuralNetworkModel;

    public GradientDescentWithL2RegularizationProcessProvider(double regularizationCoeff, NeuralNetworkModel originalNeuralNetworkModel) {
        this.regularizationCoeff = regularizationCoeff;
        this.originalNeuralNetworkModel = originalNeuralNetworkModel.clone();
        this.gradientDescentProcessProvider = new GradientDescentProcessProvider();
    }

    @Override
    public Function<GradientDescentCorrectionsContainer, GradientDescentCorrectionsContainer> getGradientDescentCorrectionsLauncher() {
        return gradientDescentProcessProvider.getGradientDescentCorrectionsLauncher()
                .andThen(gradientDescentCorrectionsContainer -> {
                    NeuralNetworkModel neuralNetworkModel = gradientDescentCorrectionsContainer.getCorrectedNeuralNetworkModel();
                    List<Layer> layers = neuralNetworkModel.getLayers();
                    List<Layer> originalLayers = originalNeuralNetworkModel.getLayers();
                    for (int layerIndex = 0; layerIndex < layers.size(); layerIndex++) {
                        Layer layer = layers.get(layerIndex);
                        Layer originalLayer = originalLayers.get(layerIndex);
                        layer.getWeightMatrix().subi(originalLayer.getWeightMatrix().mul(gradientDescentCorrectionsContainer.getLearningRate() * regularizationCoeff / gradientDescentCorrectionsContainer.getInputCount()));
                    }
                    return new GradientDescentCorrectionsContainer(neuralNetworkModel, gradientDescentCorrectionsContainer.getGradientDescentCorrections(), gradientDescentCorrectionsContainer.getInputCount(), gradientDescentCorrectionsContainer.getLearningRate());
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
    public Function<ForwardComputationContainer, GradientLayerProvider> getForwardComputationLauncher() {
        return gradientDescentProcessProvider.getForwardComputationLauncher();
    }

    @Override
    public Function<DataContainer, DataContainer> getDataProcessLauncher() {
        return Function.identity();
    }
}
