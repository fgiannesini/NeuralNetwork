package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescentwithderivation;

import com.fgiannesini.neuralnetwork.cost.CostType;
import com.fgiannesini.neuralnetwork.learningalgorithm.LearningAlgorithm;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.DataContainer;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.GradientDescentCorrection;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import org.jblas.DoubleMatrix;

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
    public NeuralNetworkModel learn(DoubleMatrix inputMatrix, DoubleMatrix y) {
        DataContainer dataContainer = gradientDescentProcessProvider.getDataProcessLauncher().apply(new DataContainer(inputMatrix, y));
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
