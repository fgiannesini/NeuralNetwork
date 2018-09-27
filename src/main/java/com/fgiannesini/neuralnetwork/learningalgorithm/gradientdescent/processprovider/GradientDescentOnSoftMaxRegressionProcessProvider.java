package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.processprovider;

import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.container.ErrorComputationContainer;

import java.util.function.Function;

public class GradientDescentOnSoftMaxRegressionProcessProvider implements IGradientDescentProcessProvider {

    private IGradientDescentProcessProvider processProvider;

    public GradientDescentOnSoftMaxRegressionProcessProvider(IGradientDescentProcessProvider processProvider) {
        this.processProvider = processProvider;
    }

    @Override
    public IGradientDescentProcessProvider getPreviousProcessProvider() {
        return processProvider;
    }

    @Override
    public Function<ErrorComputationContainer, ErrorComputationContainer> getFirstErrorComputationLauncher() {
        return container -> {
            GradientDescentSoftMaxRegressionVisitor softMaxRegressionVisitor = new GradientDescentSoftMaxRegressionVisitor(container.getProvider());
            container.getPreviousError().accept(softMaxRegressionVisitor);
            return new ErrorComputationContainer(container.getProvider(), softMaxRegressionVisitor.getErrorData(), container.getCurrentLayerIndex());
        };
    }

}
