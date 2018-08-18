package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent;

import org.jblas.DoubleMatrix;

import java.util.List;
import java.util.function.Function;

public class GradientDescentOnSoftMaxRegressionProcessProvider implements IGradientDescentProcessProvider {

    private IGradientDescentProcessProvider processProvider;

    public GradientDescentOnSoftMaxRegressionProcessProvider() {
        processProvider = new GradientDescentDefaultProcessProvider();
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
//            DoubleMatrix error = container.getPreviousError().div(container.getProvider().getCurrentResult())
//                    .muli(-1)
//                    .muli(container.getProvider().getCurrentActivationFunction().derivate(container.getProvider().getCurrentResult()));

            //dZ2 = A2-Y .* g2'(A2)
            DoubleMatrix error = container.getProvider().getCurrentResult().sub(container.getPreviousError())
                    .muli(container.getProvider().getCurrentActivationFunction().derivate(container.getProvider().getCurrentResult()));
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
