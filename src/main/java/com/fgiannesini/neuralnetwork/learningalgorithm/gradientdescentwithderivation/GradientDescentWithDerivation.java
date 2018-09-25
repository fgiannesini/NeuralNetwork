package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescentwithderivation;

import com.fgiannesini.neuralnetwork.computer.LayerTypeData;
import com.fgiannesini.neuralnetwork.cost.CostType;
import com.fgiannesini.neuralnetwork.learningalgorithm.LearningAlgorithm;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.container.DataContainer;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.container.GradientDescentCorrection;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescentwithderivation.container.GradientDescentWithDerivationContainer;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescentwithderivation.container.GradientDescentWithDerivationCorrectionsContainer;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescentwithderivation.processprovider.IGradientDescentWithDerivationProcessProvider;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;

import java.util.List;

public class GradientDescentWithDerivation implements LearningAlgorithm {

    private NeuralNetworkModel originalNeuralNetworkModel;
    private final CostType costType;
    private double learningRate;
    private final IGradientDescentWithDerivationProcessProvider gradientDescentProcessProvider;

    public GradientDescentWithDerivation(NeuralNetworkModel neuralNetworkModel, CostType costType, IGradientDescentWithDerivationProcessProvider gradientDescentProcessProvider) {
        this.originalNeuralNetworkModel = neuralNetworkModel.clone();
        this.costType = costType;
        this.learningRate = 0.01;
        this.gradientDescentProcessProvider = gradientDescentProcessProvider;
    }

    @Override
    public NeuralNetworkModel learn(LayerTypeData inputData, LayerTypeData outputData) {
        DataContainer dataContainer = gradientDescentProcessProvider.getDataProcessLauncher().apply(new DataContainer(inputData, outputData));
        GradientDescentWithDerivationContainer gradientDescentWithDerivationContainer = new GradientDescentWithDerivationContainer(dataContainer.getInput(), dataContainer.getOutput(), originalNeuralNetworkModel, costType, gradientDescentProcessProvider.getCostComputerBuildingLauncher());
        List<GradientDescentCorrection> gradientDescentCorrections = gradientDescentProcessProvider.getGradientWithDerivationLauncher().apply(gradientDescentWithDerivationContainer);
        originalNeuralNetworkModel = gradientDescentProcessProvider.getGradientDescentCorrectionsLauncher().apply(new GradientDescentWithDerivationCorrectionsContainer(originalNeuralNetworkModel, gradientDescentCorrections, dataContainer.getOutput().getColumns(), learningRate)).getCorrectedNeuralNetworkModel();
        return originalNeuralNetworkModel;
    }

    @Override
    public void updateLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    public IGradientDescentWithDerivationProcessProvider getGradientDescentProcessProvider() {
        return gradientDescentProcessProvider;
    }
}
