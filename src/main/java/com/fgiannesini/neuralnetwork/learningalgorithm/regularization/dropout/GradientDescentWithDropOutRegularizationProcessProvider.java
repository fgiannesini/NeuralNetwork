package com.fgiannesini.neuralnetwork.learningalgorithm.regularization.dropout;

import com.fgiannesini.neuralnetwork.computer.OutputComputerBuilder;
import com.fgiannesini.neuralnetwork.computer.intermediateoutputcomputer.IIntermediateOutputComputer;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.*;
import com.fgiannesini.neuralnetwork.model.Layer;
import org.jblas.DoubleMatrix;

import java.util.List;
import java.util.function.Function;
import java.util.function.Supplier;

public class GradientDescentWithDropOutRegularizationProcessProvider implements IGradientDescentProcessProvider {

    private final IGradientDescentProcessProvider gradientDescentProcessProvider;
    private final Supplier<List<DoubleMatrix>> dropOutMatrices;

    public GradientDescentWithDropOutRegularizationProcessProvider(Supplier<List<DoubleMatrix>> dropOutMatrices) {
        this.dropOutMatrices = dropOutMatrices;
        gradientDescentProcessProvider = new GradientDescentProcessProvider();
    }

    @Override
    public Function<ErrorComputationContainer, ErrorComputationContainer> getErrorComputationLauncher() {
        return gradientDescentProcessProvider.getErrorComputationLauncher().andThen(container -> {
            DoubleMatrix dropOutMatrix = dropOutMatrices.get().get(container.getProvider().getCurrentLayerIndex());
            DoubleMatrix error = container.getPreviousError().muliColumnVector(dropOutMatrix);
            return new ErrorComputationContainer(container.getProvider(), error);
        });
    }

    @Override
    public Function<ErrorComputationContainer, ErrorComputationContainer> getFirstErrorComputationLauncher() {
        return gradientDescentProcessProvider.getFirstErrorComputationLauncher().andThen(container -> {
            DoubleMatrix dropOutMatrix = dropOutMatrices.get().get(container.getProvider().getCurrentLayerIndex());
            DoubleMatrix error = container.getPreviousError().muliColumnVector(dropOutMatrix);
            return new ErrorComputationContainer(container.getProvider(), error);
        });
    }

    @Override
    public Function<GradientDescentCorrectionsContainer, GradientDescentCorrectionsContainer> getGradientDescentCorrectionsLauncher() {
        return gradientDescentProcessProvider.getGradientDescentCorrectionsLauncher();
    }

    @Override
    public Function<BackwardComputationContainer, List<GradientDescentCorrection>> getBackwardComputationLauncher() {
        return gradientDescentProcessProvider.getBackwardComputationLauncher();
    }

    @Override
    public Function<ForwardComputationContainer, GradientLayerProvider> getForwardComputationLauncher() {
        return container -> {
            List<Layer> layers = container.getNeuralNetworkModel().getLayers();
            IIntermediateOutputComputer intermediateOutputComputer = OutputComputerBuilder.init()
                    .withModel(container.getNeuralNetworkModel())
                    .withDropOutParameters(dropOutMatrices.get())
                    .buildIntermediateOutputComputer();
            List<DoubleMatrix> intermediateResults = intermediateOutputComputer.compute(container.getInputMatrix());
            return new GradientLayerProvider(layers, intermediateResults);
        };
    }
}
