package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.processprovider;

import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.container.ErrorComputationContainer;

import java.util.function.Function;

public class GradientDescentOnLinearRegressionProcessProvider implements IGradientDescentProcessProvider {

    private final IGradientDescentProcessProvider processProvider;

    public GradientDescentOnLinearRegressionProcessProvider(IGradientDescentProcessProvider processProvider) {
        this.processProvider = processProvider;
    }

    public IGradientDescentProcessProvider getPreviousProcessProvider() {
        return processProvider;
    }

    @Override
    public Function<ErrorComputationContainer, ErrorComputationContainer> getFirstErrorComputationLauncher() {
        return container -> {
            GradientDescentLinearRegressionVisitor linearRegressionVisitor = new GradientDescentLinearRegressionVisitor(container.getProvider());
            container.getPreviousError().accept(linearRegressionVisitor);
            return new ErrorComputationContainer(container.getProvider(), linearRegressionVisitor.getErrorData());
        };
    }

}
