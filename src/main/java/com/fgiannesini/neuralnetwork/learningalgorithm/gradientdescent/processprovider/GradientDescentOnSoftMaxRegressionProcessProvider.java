package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.processprovider;

import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.container.ErrorComputationContainer;
import org.jblas.DoubleMatrix;

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
            //dZ2 = A2-Y .* g2'(A2)
            DoubleMatrix error = container.getProvider().getCurrentResult().sub(container.getPreviousError())
                    .muli(container.getProvider().getActivationFunction().derivate(container.getProvider().getCurrentResult()));
            return new ErrorComputationContainer(container.getProvider(), error);
        };
    }

}
