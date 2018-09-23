package com.fgiannesini.neuralnetwork.learningalgorithm.regularization.l2;

import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.container.DataContainer;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.container.GradientDescentCorrectionsContainer;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.processprovider.IGradientDescentProcessProvider;
import com.fgiannesini.neuralnetwork.model.Layer;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;

import java.util.List;
import java.util.function.Function;

public class GradientDescentWithL2RegularizationProcessProvider implements IGradientDescentProcessProvider {

    private final IGradientDescentProcessProvider gradientDescentProcessProvider;
    private final double regularizationCoeff;
    private final NeuralNetworkModel originalNeuralNetworkModel;

    public GradientDescentWithL2RegularizationProcessProvider(double regularizationCoeff, NeuralNetworkModel originalNeuralNetworkModel, IGradientDescentProcessProvider gradientDescentProcessProvider) {
        this.regularizationCoeff = regularizationCoeff;
        this.originalNeuralNetworkModel = originalNeuralNetworkModel.clone();
        this.gradientDescentProcessProvider = gradientDescentProcessProvider;
    }

    @Override
    public IGradientDescentProcessProvider getPreviousProcessProvider() {
        return gradientDescentProcessProvider;
    }

    @Override
    public Function<GradientDescentCorrectionsContainer, GradientDescentCorrectionsContainer> getGradientDescentCorrectionsLauncher() {
        return gradientDescentProcessProvider.getGradientDescentCorrectionsLauncher()
                .andThen(gradientDescentCorrectionsContainer -> {
                    NeuralNetworkModel neuralNetworkModel = gradientDescentCorrectionsContainer.getCorrectedNeuralNetworkModel();
                    List<Layer> layers = neuralNetworkModel.getLayers();
                    List<Layer> originalLayers = originalNeuralNetworkModel.getLayers();
                    for (int layerIndex = 0; layerIndex < layers.size(); layerIndex++) {
                        GradientDescentWithL2RegularizationVisitor l2RegularizationVisitor = new GradientDescentWithL2RegularizationVisitor(originalLayers.get(layerIndex), regularizationCoeff, gradientDescentCorrectionsContainer.getLearningRate(), gradientDescentCorrectionsContainer.getInputCount());
                        layers.get(layerIndex).accept(l2RegularizationVisitor);
                    }
                    return new GradientDescentCorrectionsContainer(neuralNetworkModel, gradientDescentCorrectionsContainer.getGradientDescentCorrections(), gradientDescentCorrectionsContainer.getInputCount(), gradientDescentCorrectionsContainer.getLearningRate());
                });
    }

    @Override
    public Function<DataContainer, DataContainer> getDataProcessLauncher() {
        return Function.identity();
    }
}
