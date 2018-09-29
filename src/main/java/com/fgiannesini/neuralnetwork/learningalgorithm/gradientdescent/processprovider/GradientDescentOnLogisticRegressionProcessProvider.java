package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.processprovider;

import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.container.ErrorComputationContainer;

import java.util.function.Function;

public class GradientDescentOnLogisticRegressionProcessProvider implements IGradientDescentProcessProvider {

    private final IGradientDescentProcessProvider processProvider;

    public GradientDescentOnLogisticRegressionProcessProvider(IGradientDescentProcessProvider processProvider) {
        this.processProvider = processProvider;
    }

    @Override
    public IGradientDescentProcessProvider getPreviousProcessProvider() {
        return processProvider;
    }

    @Override
    public Function<ErrorComputationContainer, ErrorComputationContainer> getFirstErrorComputationLauncher() {
        return container -> {
            GradientDescentLogisticRegressionVisitor logisticRegressionVisitor = new GradientDescentLogisticRegressionVisitor(container.getProvider());
            container.getPreviousError().accept(logisticRegressionVisitor);
            return new ErrorComputationContainer(container.getProvider(), logisticRegressionVisitor.getErrorData());
        };
    }

}
