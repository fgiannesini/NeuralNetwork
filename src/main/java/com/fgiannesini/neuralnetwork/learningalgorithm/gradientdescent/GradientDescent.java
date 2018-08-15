package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent;

import com.fgiannesini.neuralnetwork.learningalgorithm.LearningAlgorithm;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import org.jblas.DoubleMatrix;

import java.util.List;

public class GradientDescent implements LearningAlgorithm {
    private NeuralNetworkModel correctedNeuralNetworkModel;
    private double learningRate;
    private final IGradientDescentProcessProvider gradientDescentProcessProvider;

    public GradientDescent(NeuralNetworkModel originalNeuralNetworkModel, IGradientDescentProcessProvider gradientDescentProcessProvider) {
        this.gradientDescentProcessProvider = gradientDescentProcessProvider;
        this.correctedNeuralNetworkModel = originalNeuralNetworkModel.clone();
        this.learningRate = 0.01;
    }

    @Override
    public NeuralNetworkModel learn(DoubleMatrix inputMatrix, DoubleMatrix y) {
        DataContainer dataContainer = new DataContainer(inputMatrix, y);
        dataContainer = gradientDescentProcessProvider.getDataProcessLauncher().apply(dataContainer);

        GradientLayerProvider provider = gradientDescentProcessProvider.getForwardComputationLauncher()
                .apply(new ForwardComputationContainer(dataContainer.getInput(), correctedNeuralNetworkModel));

        List<GradientDescentCorrection> gradientDescentCorrections = gradientDescentProcessProvider.getBackwardComputationLauncher()
                .apply(new BackwardComputationContainer(provider, dataContainer.getOutput(), gradientDescentProcessProvider.getFirstErrorComputationLauncher(), gradientDescentProcessProvider.getErrorComputationLauncher()));

        correctedNeuralNetworkModel = gradientDescentProcessProvider.getGradientDescentCorrectionsLauncher()
                .apply(new GradientDescentCorrectionsContainer(this.correctedNeuralNetworkModel, gradientDescentCorrections, dataContainer.getOutput().getColumns(), learningRate))
                .getCorrectedNeuralNetworkModel();
        return correctedNeuralNetworkModel;
    }

    @Override
    public void updateLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    public IGradientDescentProcessProvider getGradientDescentProcessProvider() {
        return gradientDescentProcessProvider;
    }
}
