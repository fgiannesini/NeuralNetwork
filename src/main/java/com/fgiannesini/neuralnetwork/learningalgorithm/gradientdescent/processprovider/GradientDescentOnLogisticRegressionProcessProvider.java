package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.processprovider;

import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.container.ErrorComputationContainer;
import org.jblas.DoubleMatrix;

import java.util.function.Function;

public class GradientDescentOnLogisticRegressionProcessProvider implements IGradientDescentProcessProvider {

    private IGradientDescentProcessProvider processProvider;

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
            //dZ2 = (A2 - Y)/A2(1-A2)) .* g2'(A2)
            DoubleMatrix error = container.getProvider().getCurrentResult().sub(container.getPreviousError())
                    .divi(container.getProvider().getCurrentResult())
                    .divi(container.getProvider().getCurrentResult().neg().addi(1))
                    .muli(container.getProvider().getCurrentActivationFunction().derivate(container.getProvider().getCurrentResult()));

            return new ErrorComputationContainer(container.getProvider(), error);
        };
    }

}
