package com.fgiannesini.neuralnetwork.learningalgorithm.regularization.dropout;

import com.fgiannesini.neuralnetwork.computer.OutputComputerBuilder;
import com.fgiannesini.neuralnetwork.computer.intermediateoutputcomputer.IIntermediateOutputComputer;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.GradientLayerProvider;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.container.*;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.processprovider.IGradientDescentProcessProvider;
import com.fgiannesini.neuralnetwork.model.Layer;
import org.jblas.DoubleMatrix;

import java.util.List;
import java.util.function.Function;
import java.util.function.Supplier;

public class GradientDescentWithDropOutRegularizationProcessProvider<L extends Layer> implements IGradientDescentProcessProvider<L> {

    private final IGradientDescentProcessProvider<L> gradientDescentProcessProvider;
    private final Supplier<List<DoubleMatrix>> dropOutMatricesSupplier;
    private List<DoubleMatrix> dropOutMatrices;

    public GradientDescentWithDropOutRegularizationProcessProvider(Supplier<List<DoubleMatrix>> dropOutMatricesSupplier, IGradientDescentProcessProvider<L> processProvider) {
        this.dropOutMatricesSupplier = dropOutMatricesSupplier;
        gradientDescentProcessProvider = processProvider;
    }

    @Override
    public Function<ErrorComputationContainer, ErrorComputationContainer> getErrorComputationLauncher() {
        return gradientDescentProcessProvider.getErrorComputationLauncher().andThen(container -> {
            DoubleMatrix dropOutMatrix = dropOutMatrices.get(container.getProvider().getCurrentLayerIndex());
            DoubleMatrix error = container.getPreviousError().muliColumnVector(dropOutMatrix);
            return new ErrorComputationContainer(container.getProvider(), error);
        });
    }

    @Override
    public Function<ErrorComputationContainer, ErrorComputationContainer> getFirstErrorComputationLauncher() {
        return gradientDescentProcessProvider.getFirstErrorComputationLauncher().andThen(container -> {
            DoubleMatrix dropOutMatrix = dropOutMatrices.get(container.getProvider().getCurrentLayerIndex());
            DoubleMatrix error = container.getPreviousError().muliColumnVector(dropOutMatrix);
            return new ErrorComputationContainer(container.getProvider(), error);
        });
    }

    @Override
    public Function<GradientDescentCorrectionsContainer<L>, GradientDescentCorrectionsContainer<L>> getGradientDescentCorrectionsLauncher() {
        return gradientDescentProcessProvider.getGradientDescentCorrectionsLauncher();
    }

    @Override
    public Function<BackwardComputationContainer, List<GradientDescentCorrection>> getBackwardComputationLauncher() {
        return gradientDescentProcessProvider.getBackwardComputationLauncher();
    }

    @Override
    public Function<ForwardComputationContainer<L>, GradientLayerProvider<L>> getForwardComputationLauncher() {
        return container -> {
            List<L> layers = container.getNeuralNetworkModel().getLayers();
            IIntermediateOutputComputer intermediateOutputComputer = OutputComputerBuilder.init()
                    .withModel(container.getNeuralNetworkModel())
                    .withDropOutParameters(dropOutMatrices)
                    .buildIntermediateOutputComputer();
            List<DoubleMatrix> intermediateResults = intermediateOutputComputer.compute(container.getInputMatrix());
            return new GradientLayerProvider<>(layers, intermediateResults);
        };
    }

    @Override
    public Function<DataContainer, DataContainer> getDataProcessLauncher() {
        return container-> {
            dropOutMatrices = dropOutMatricesSupplier.get();
            DoubleMatrix dropOutOutput = container.getOutput().mulColumnVector(dropOutMatrices.get(dropOutMatrices.size() - 1));
            return new DataContainer(container.getInput(), dropOutOutput);
        };
    }
}
