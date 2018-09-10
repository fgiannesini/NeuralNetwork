package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.processprovider;

import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.container.*;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.layerdataprovider.GradientLayerProvider;
import org.jblas.DoubleMatrix;

import java.util.List;
import java.util.function.Function;

public class GradientDescentOnSoftMaxRegressionProcessProvider implements IGradientDescentProcessProvider {

    private IGradientDescentProcessProvider processProvider;

    public GradientDescentOnSoftMaxRegressionProcessProvider(IGradientDescentProcessProvider processProvider) {
        this.processProvider = processProvider;
    }

    @Override
    public Function<GradientDescentCorrectionsContainer, GradientDescentCorrectionsContainer> getGradientDescentCorrectionsLauncher() {
        return processProvider.getGradientDescentCorrectionsLauncher();
    }

    @Override
    public Function<BackwardComputationContainer, List<GradientDescentCorrection>> getBackwardComputationLauncher() {
        return processProvider.getBackwardComputationLauncher();
    }

    @Override
    public Function<ErrorComputationContainer, ErrorComputationContainer> getErrorComputationLauncher() {
        return processProvider.getErrorComputationLauncher();
    }

    @Override
    public Function<ErrorComputationContainer, ErrorComputationContainer> getFirstErrorComputationLauncher() {
        return container -> {
            //dZ2 = -Y/A2 .* g2'(A2)
            DoubleMatrix delta = container.getPreviousError().mul(-1).divi(container.getProvider().getCurrentResult());
            DoubleMatrix error = container.getProvider().getCurrentActivationFunction().derivate(container.getProvider().getCurrentResult(), delta);
            return new ErrorComputationContainer(container.getProvider(), error);
        };
    }

    @Override
    public Function<ForwardComputationContainer, GradientLayerProvider> getForwardComputationLauncher() {
        return processProvider.getForwardComputationLauncher();
    }

    @Override
    public Function<DataContainer, DataContainer> getDataProcessLauncher() {
        return processProvider.getDataProcessLauncher();
    }
}
