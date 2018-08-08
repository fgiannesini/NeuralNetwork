package com.fgiannesini.neuralnetwork.learningalgorithm.regularization.dropout;

import com.fgiannesini.neuralnetwork.cost.CostComputer;
import com.fgiannesini.neuralnetwork.cost.CostComputerBuilder;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.DataContainer;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescentwithderivation.GradientDescentWithDerivationContainer;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescentwithderivation.GradientDescentWithDerivationCostComputerContainer;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescentwithderivation.IGradientDescentWithDerivationProcessProvider;
import org.jblas.DoubleMatrix;

import java.util.List;
import java.util.function.Function;
import java.util.function.Supplier;

public class GradientDescentWithDerivationAndDropOutRegularizationProcessProvider implements IGradientDescentWithDerivationProcessProvider {

    private final IGradientDescentWithDerivationProcessProvider gradientDescentWithDerivationProcessProvider;
    private List<DoubleMatrix> dropOutMatrices;
    private final Supplier<List<DoubleMatrix>> dropOutMatricesSupplier;

    public GradientDescentWithDerivationAndDropOutRegularizationProcessProvider(Supplier<List<DoubleMatrix>> dropOutMatricesSupplier, IGradientDescentWithDerivationProcessProvider gradientDescentWithDerivationProcessProvider) {
        this.dropOutMatricesSupplier = dropOutMatricesSupplier;
        this.gradientDescentWithDerivationProcessProvider = gradientDescentWithDerivationProcessProvider;
    }

    @Override
    public Function<DataContainer, DataContainer> getDataProcessLauncher() {
        return container -> {
            dropOutMatrices = dropOutMatricesSupplier.get();
            DoubleMatrix dropOutOutput = container.getOutput().mulColumnVector(dropOutMatrices.get(dropOutMatrices.size() - 1));
            return new DataContainer(container.getInput(), dropOutOutput);
        };
    }

    @Override
    public Function<GradientDescentWithDerivationContainer, GradientDescentWithDerivationContainer> getGradientWithDerivationLauncher() {
        return gradientDescentWithDerivationProcessProvider.getGradientWithDerivationLauncher();
    }

    @Override
    public Function<GradientDescentWithDerivationCostComputerContainer, CostComputer> getCostComputerBuildingLauncher() {
        return container -> CostComputerBuilder.init()
                .withNeuralNetworkModel(container.getNeuralNetworkModel())
                .withDropOutRegularization(dropOutMatrices)
                .withType(container.getCostType())
                .build();
    }
}
