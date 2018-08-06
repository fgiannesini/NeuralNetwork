package com.fgiannesini.neuralnetwork.learningalgorithm.regularization.dropout;

import com.fgiannesini.neuralnetwork.cost.CostComputer;
import com.fgiannesini.neuralnetwork.cost.CostComputerBuilder;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescentwithderivation.GradientDescentWithDerivationContainer;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescentwithderivation.GradientDescentWithDerivationCostComputerContainer;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescentwithderivation.GradientDescentWithDerivationProcessProvider;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescentwithderivation.IGradientDescentWithDerivationProcessProvider;
import org.jblas.DoubleMatrix;

import java.util.List;
import java.util.function.Function;

public class GradientDescentWithDerivationandDropOutRegularizationProcessProvider implements IGradientDescentWithDerivationProcessProvider {

    private final IGradientDescentWithDerivationProcessProvider gradientDescentWithDerivationProcessProvider;
    private final List<DoubleMatrix> dropOutMatrices;

    public GradientDescentWithDerivationandDropOutRegularizationProcessProvider(List<DoubleMatrix> dropOutMatrices) {
        this.dropOutMatrices = dropOutMatrices;
        gradientDescentWithDerivationProcessProvider = new GradientDescentWithDerivationProcessProvider();
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
