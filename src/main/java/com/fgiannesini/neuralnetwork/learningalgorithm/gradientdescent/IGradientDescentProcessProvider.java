package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent;

import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;

import java.util.List;
import java.util.function.Function;

public interface IGradientDescentProcessProvider {

    Function<GradientDescentCorrectionsContainer, NeuralNetworkModel> getGradientDescentCorrectionsLauncher();

    Function<BackwardComputationContainer, List<GradientDescentCorrection>> getBackwardComputationLauncher();

    Function<ErrorComputationContainer, ErrorComputationContainer> getErrorComputationLauncher();

    Function<ErrorComputationContainer, ErrorComputationContainer> getFirstErrorComputationLauncher();

    Function<ForwardComputationContainer, GradientLayerProvider> getForwardComputationLauncher();
}
