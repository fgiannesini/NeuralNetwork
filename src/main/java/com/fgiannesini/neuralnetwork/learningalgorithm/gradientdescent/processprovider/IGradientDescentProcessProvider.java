package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.processprovider;

import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.container.*;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.layerdataprovider.GradientLayerProvider;

import java.util.List;
import java.util.function.Function;

public interface IGradientDescentProcessProvider {

    IGradientDescentProcessProvider getPreviousProcessProvider();

    default Function<GradientDescentCorrectionsContainer, GradientDescentCorrectionsContainer> getGradientDescentCorrectionsLauncher() {
        return getPreviousProcessProvider().getGradientDescentCorrectionsLauncher();
    }

    default Function<BackwardComputationContainer, List<GradientDescentCorrection>> getBackwardComputationLauncher() {
        return getPreviousProcessProvider().getBackwardComputationLauncher();
    }

    default Function<ErrorComputationContainer, ErrorComputationContainer> getErrorComputationLauncher() {
        return getPreviousProcessProvider().getErrorComputationLauncher();
    }

    default Function<ErrorComputationContainer, ErrorComputationContainer> getFirstErrorComputationLauncher() {
        return getPreviousProcessProvider().getFirstErrorComputationLauncher();
    }

    default Function<ForwardComputationContainer, GradientLayerProvider> getForwardComputationLauncher() {
        return getPreviousProcessProvider().getForwardComputationLauncher();
    }

    default Function<DataContainer, DataContainer> getDataProcessLauncher() {
        return getPreviousProcessProvider().getDataProcessLauncher();
    }
}
