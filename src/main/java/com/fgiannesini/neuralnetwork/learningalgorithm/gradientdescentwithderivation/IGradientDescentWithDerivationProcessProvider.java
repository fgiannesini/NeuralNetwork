package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescentwithderivation;

import com.fgiannesini.neuralnetwork.cost.CostComputer;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.DataContainer;

import java.util.function.Function;

public interface IGradientDescentWithDerivationProcessProvider {

    Function<DataContainer, DataContainer> getDataProcessLauncher();

    Function<GradientDescentWithDerivationContainer, GradientDescentWithDerivationContainer> getGradientWithDerivationLauncher();

    Function<GradientDescentWithDerivationCostComputerContainer, CostComputer> getCostComputerBuildingLauncher();
}
