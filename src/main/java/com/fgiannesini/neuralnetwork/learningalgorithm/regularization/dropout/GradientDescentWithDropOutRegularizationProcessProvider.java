package com.fgiannesini.neuralnetwork.learningalgorithm.regularization.dropout;

import com.fgiannesini.neuralnetwork.computer.OutputComputerBuilder;
import com.fgiannesini.neuralnetwork.computer.intermediateoutputcomputer.IIntermediateOutputComputer;
import com.fgiannesini.neuralnetwork.computer.intermediateoutputcomputer.IntermediateOutputResult;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.container.DataContainer;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.container.ErrorComputationContainer;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.container.ForwardComputationContainer;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.layerdataprovider.GradientLayerProvider;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.layerdataprovider.GradientLayerProviderBuilder;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.processprovider.IGradientDescentProcessProvider;
import org.jblas.DoubleMatrix;

import java.util.List;
import java.util.function.Function;
import java.util.function.Supplier;

public class GradientDescentWithDropOutRegularizationProcessProvider implements IGradientDescentProcessProvider {

    private final IGradientDescentProcessProvider gradientDescentProcessProvider;
    private final Supplier<List<DoubleMatrix>> dropOutMatricesSupplier;
    private List<DoubleMatrix> dropOutMatrices;

    public GradientDescentWithDropOutRegularizationProcessProvider(Supplier<List<DoubleMatrix>> dropOutMatricesSupplier, IGradientDescentProcessProvider processProvider) {
        this.dropOutMatricesSupplier = dropOutMatricesSupplier;
        gradientDescentProcessProvider = processProvider;
    }

    @Override
    public Function<ErrorComputationContainer, ErrorComputationContainer> getErrorComputationLauncher() {
        return gradientDescentProcessProvider.getErrorComputationLauncher().andThen(container -> {
            DoubleMatrix dropOutMatrix = dropOutMatrices.get(container.getProvider().getLayerIndex() + 1);
            DropOutApplierVisitor dropOutApplierVisitor = new DropOutApplierVisitor(dropOutMatrix);
            container.getPreviousError().accept(dropOutApplierVisitor);
            return new ErrorComputationContainer(container.getProvider(), dropOutApplierVisitor.getLayerTypeData());
        });
    }

    @Override
    public Function<ErrorComputationContainer, ErrorComputationContainer> getFirstErrorComputationLauncher() {
        return gradientDescentProcessProvider.getFirstErrorComputationLauncher().andThen(container -> {
            DoubleMatrix dropOutMatrix = dropOutMatrices.get(container.getProvider().getLayerIndex() + 1);
            DropOutApplierVisitor dropOutApplierVisitor = new DropOutApplierVisitor(dropOutMatrix);
            container.getPreviousError().accept(dropOutApplierVisitor);
            return new ErrorComputationContainer(container.getProvider(), dropOutApplierVisitor.getLayerTypeData());
        });
    }

    @Override
    public IGradientDescentProcessProvider getPreviousProcessProvider() {
        return gradientDescentProcessProvider;
    }

    @Override
    public Function<ForwardComputationContainer, List<GradientLayerProvider>> getForwardComputationLauncher() {
        return container -> {
            IIntermediateOutputComputer intermediateOutputComputer = OutputComputerBuilder.init()
                    .withModel(container.getNeuralNetworkModel())
                    .withDropOutParameters(dropOutMatrices)
                    .buildIntermediateOutputComputer();

            List<IntermediateOutputResult> intermediateResults = intermediateOutputComputer.compute(container.getInput());
            return GradientLayerProviderBuilder.init()
                    .withModel(container.getNeuralNetworkModel())
                    .withIntermediateResults(intermediateResults)
                    .build();
        };
    }

    @Override
    public Function<DataContainer, DataContainer> getDataProcessLauncher() {
        return container -> {
            dropOutMatrices = dropOutMatricesSupplier.get();
            DoubleMatrix dropOutMatrix = dropOutMatrices.get(dropOutMatrices.size() - 1);
            DropOutApplierVisitor dropOutApplierVisitor = new DropOutApplierVisitor(dropOutMatrix);
            container.getOutput().accept(dropOutApplierVisitor);
            return new DataContainer(container.getInput(), dropOutApplierVisitor.getLayerTypeData());
        };
    }
}
