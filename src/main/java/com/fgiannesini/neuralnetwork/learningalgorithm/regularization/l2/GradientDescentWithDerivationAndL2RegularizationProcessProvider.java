package com.fgiannesini.neuralnetwork.learningalgorithm.regularization.l2;

import com.fgiannesini.neuralnetwork.cost.CostComputer;
import com.fgiannesini.neuralnetwork.cost.CostComputerBuilder;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.container.DataContainer;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.container.GradientDescentCorrection;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescentwithderivation.container.GradientDescentWithDerivationContainer;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescentwithderivation.container.GradientDescentWithDerivationCorrectionsContainer;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescentwithderivation.container.GradientDescentWithDerivationCostComputerContainer;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescentwithderivation.processprovider.IGradientDescentWithDerivationProcessProvider;

import java.util.List;
import java.util.function.Function;

public class GradientDescentWithDerivationAndL2RegularizationProcessProvider implements IGradientDescentWithDerivationProcessProvider {

    private final double regularizationCoeff;
    private final IGradientDescentWithDerivationProcessProvider gradientDescentProcessProvider;

    public GradientDescentWithDerivationAndL2RegularizationProcessProvider(double regularizationCoeff, IGradientDescentWithDerivationProcessProvider gradientDescentProcessProvider) {
        this.regularizationCoeff = regularizationCoeff;
        this.gradientDescentProcessProvider = gradientDescentProcessProvider;
    }

    @Override
    public Function<DataContainer, DataContainer> getDataProcessLauncher() {
        return Function.identity();
    }

    @Override
    public Function<GradientDescentWithDerivationContainer, List<GradientDescentCorrection>> getGradientWithDerivationLauncher() {
        return gradientDescentProcessProvider.getGradientWithDerivationLauncher();
    }

    @Override
    public Function<GradientDescentWithDerivationCostComputerContainer, CostComputer> getCostComputerBuildingLauncher() {
        return container-> CostComputerBuilder.init()
                .withNeuralNetworkModel(container.getNeuralNetworkModel())
                .withType(container.getCostType())
                .withL2Regularization(regularizationCoeff)
                .build();
    }

    @Override
    public Function<GradientDescentWithDerivationCorrectionsContainer, GradientDescentWithDerivationCorrectionsContainer> getGradientDescentCorrectionsLauncher() {
        return gradientDescentProcessProvider.getGradientDescentCorrectionsLauncher();
    }
}
