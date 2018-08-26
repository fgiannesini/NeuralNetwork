package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescentwithderivation.processprovider;

import com.fgiannesini.neuralnetwork.cost.CostComputer;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.container.DataContainer;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.container.GradientDescentCorrection;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescentwithderivation.container.GradientDescentWithDerivationContainer;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescentwithderivation.container.GradientDescentWithDerivationCorrectionsContainer;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescentwithderivation.container.GradientDescentWithDerivationCostComputerContainer;

import java.util.List;
import java.util.function.Function;

public interface IGradientDescentWithDerivationProcessProvider {

    Function<DataContainer, DataContainer> getDataProcessLauncher();

    Function<GradientDescentWithDerivationContainer, List<GradientDescentCorrection>> getGradientWithDerivationLauncher();

    Function<GradientDescentWithDerivationCostComputerContainer, CostComputer> getCostComputerBuildingLauncher();

    Function<GradientDescentWithDerivationCorrectionsContainer, GradientDescentWithDerivationCorrectionsContainer> getGradientDescentCorrectionsLauncher();
}
