package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.container;

import com.fgiannesini.neuralnetwork.computer.LayerTypeData;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.layerdataprovider.GradientLayerProvider;

import java.util.function.Function;

public class BackwardComputationContainer {
    private final GradientLayerProvider provider;
    private final LayerTypeData y;
    private final Function<ErrorComputationContainer, ErrorComputationContainer> firstErrorComputationLauncher;
    private final Function<ErrorComputationContainer, ErrorComputationContainer> errorComputationLauncher;

    public BackwardComputationContainer(GradientLayerProvider provider, LayerTypeData output, Function<ErrorComputationContainer, ErrorComputationContainer> firstErrorComputationLauncher, Function<ErrorComputationContainer, ErrorComputationContainer> errorComputationLauncher) {
        this.provider = provider;
        this.y = output;
        this.firstErrorComputationLauncher = firstErrorComputationLauncher;
        this.errorComputationLauncher = errorComputationLauncher;
    }

    public GradientLayerProvider getProvider() {
        return provider;
    }

    public LayerTypeData getY() {
        return y;
    }

    public Function<ErrorComputationContainer, ErrorComputationContainer> getFirstErrorComputationLauncher() {
        return firstErrorComputationLauncher;
    }

    public Function<ErrorComputationContainer, ErrorComputationContainer> getErrorComputationLauncher() {
        return errorComputationLauncher;
    }
}
