package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescentwithderivation;

import com.fgiannesini.neuralnetwork.cost.CostComputer;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.DataContainer;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.GradientDescentCorrection;

import java.util.List;
import java.util.function.Function;

public interface IGradientDescentWithDerivationProcessProvider {

    Function<DataContainer, DataContainer> getDataProcessLauncher();

    Function<GradientDescentWithDerivationContainer, List<GradientDescentCorrection>> getGradientWithDerivationLauncher();

    Function<GradientDescentWithDerivationCostComputerContainer, CostComputer> getCostComputerBuildingLauncher();

    Function<GradientDescentWithDerivationCorrectionsContainer, GradientDescentWithDerivationCorrectionsContainer> getGradientDescentCorrectionsLauncher();
}
