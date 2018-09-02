package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.processprovider;

import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.container.*;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.layerdataprovider.GradientLayerProvider;
import com.fgiannesini.neuralnetwork.model.Layer;

import java.util.List;
import java.util.function.Function;

public interface IGradientDescentProcessProvider<L extends Layer> {

    Function<GradientDescentCorrectionsContainer<L>, GradientDescentCorrectionsContainer<L>> getGradientDescentCorrectionsLauncher();

    Function<BackwardComputationContainer, List<GradientDescentCorrection>> getBackwardComputationLauncher();

    Function<ErrorComputationContainer, ErrorComputationContainer> getErrorComputationLauncher();

    Function<ErrorComputationContainer, ErrorComputationContainer> getFirstErrorComputationLauncher();

    Function<ForwardComputationContainer<L>, GradientLayerProvider<L>> getForwardComputationLauncher();

    Function<DataContainer, DataContainer> getDataProcessLauncher();
}
