package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.processprovider;

import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.container.DataContainer;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.container.ErrorComputationContainer;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.container.ForwardComputationContainer;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.container.GradientDescentCorrectionsContainer;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.layerdataprovider.GradientLayerProvider;

import java.util.List;
import java.util.function.Function;

public interface IGradientDescentProcessProvider {

    IGradientDescentProcessProvider getPreviousProcessProvider();

    default Function<GradientDescentCorrectionsContainer, GradientDescentCorrectionsContainer> getGradientDescentCorrectionsLauncher() {
        return getPreviousProcessProvider().getGradientDescentCorrectionsLauncher();
    }

    default Function<ErrorComputationContainer, ErrorComputationContainer> getErrorComputationLauncher() {
        return getPreviousProcessProvider().getErrorComputationLauncher();
    }

    default Function<ErrorComputationContainer, ErrorComputationContainer> getFirstErrorComputationLauncher() {
        return getPreviousProcessProvider().getFirstErrorComputationLauncher();
    }

    default Function<ForwardComputationContainer, List<GradientLayerProvider>> getForwardComputationLauncher() {
        return getPreviousProcessProvider().getForwardComputationLauncher();
    }

    default Function<DataContainer, DataContainer> getDataProcessLauncher() {
        return getPreviousProcessProvider().getDataProcessLauncher();
    }
}
