package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescentwithderivation;

import com.fgiannesini.neuralnetwork.cost.CostComputer;

import java.util.function.Function;

public interface IGradientDescentWithDerivationProcessProvider {
    Function<GradientDescentWithDerivationContainer, GradientDescentWithDerivationContainer> getGradientWithDerivationLauncher();

    Function<GradientDescentWithDerivationCostComputerContainer, CostComputer> getCostComputerBuildingLauncher();
}
