package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.container;

import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.GradientLayerProvider;
import org.jblas.DoubleMatrix;

import java.util.function.Function;

public class BackwardComputationContainer {
    private final GradientLayerProvider provider;
    private final DoubleMatrix y;
    private final Function<ErrorComputationContainer, ErrorComputationContainer> firstErrorComputationLauncher;
    private final Function<ErrorComputationContainer, ErrorComputationContainer> errorComputationLauncher;

    public BackwardComputationContainer(GradientLayerProvider provider, DoubleMatrix y, Function<ErrorComputationContainer, ErrorComputationContainer> firstErrorComputationLauncher, Function<ErrorComputationContainer, ErrorComputationContainer> errorComputationLauncher) {
        this.provider = provider;
        this.y = y;
        this.firstErrorComputationLauncher = firstErrorComputationLauncher;
        this.errorComputationLauncher = errorComputationLauncher;
    }

    public GradientLayerProvider getProvider() {
        return provider;
    }

    public DoubleMatrix getY() {
        return y;
    }

    public Function<ErrorComputationContainer, ErrorComputationContainer> getFirstErrorComputationLauncher() {
        return firstErrorComputationLauncher;
    }

    public Function<ErrorComputationContainer, ErrorComputationContainer> getErrorComputationLauncher() {
        return errorComputationLauncher;
    }
}
