package com.fgiannesini.neuralnetwork.learningalgorithm.regularization.dropout;

import com.fgiannesini.neuralnetwork.cost.CostComputer;
import com.fgiannesini.neuralnetwork.cost.CostComputerBuilder;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.container.DataContainer;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.container.GradientDescentCorrection;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescentwithderivation.container.GradientDescentWithDerivationContainer;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescentwithderivation.container.GradientDescentWithDerivationCorrectionsContainer;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescentwithderivation.container.GradientDescentWithDerivationCostComputerContainer;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescentwithderivation.processprovider.IGradientDescentWithDerivationProcessProvider;
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
            DropOutApplierVisitor dropOutApplierVisitor = new DropOutApplierVisitor(dropOutMatrices.get(dropOutMatrices.size() - 1));
            container.getOutput().accept(dropOutApplierVisitor);
            return new DataContainer(container.getInput(), dropOutApplierVisitor.getLayerTypeData());
        };
    }

    @Override
    public Function<GradientDescentWithDerivationContainer, List<GradientDescentCorrection>> getGradientWithDerivationLauncher() {
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

    @Override
    public Function<GradientDescentWithDerivationCorrectionsContainer, GradientDescentWithDerivationCorrectionsContainer> getGradientDescentCorrectionsLauncher() {
        return gradientDescentWithDerivationProcessProvider.getGradientDescentCorrectionsLauncher();
    }
}
