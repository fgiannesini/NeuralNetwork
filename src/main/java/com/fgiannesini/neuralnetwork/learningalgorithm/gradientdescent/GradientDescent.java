package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent;

import com.fgiannesini.neuralnetwork.learningalgorithm.LearningAlgorithm;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import org.jblas.DoubleMatrix;

import java.util.List;

public class GradientDescent implements LearningAlgorithm {
    private final NeuralNetworkModel correctedNeuralNetworkModel;
    private final double learningRate;
    private final IGradientDescentProcessProvider gradientDescentProvider;

    public GradientDescent(NeuralNetworkModel originalNeuralNetworkModel, double learningRate) {
        this.correctedNeuralNetworkModel = originalNeuralNetworkModel.clone();
        this.learningRate = learningRate;
        gradientDescentProvider = new GradientDescentProcessProvider();
    }

    @Override
    public NeuralNetworkModel learn(DoubleMatrix inputMatrix, DoubleMatrix y) {
        GradientLayerProvider provider = gradientDescentProvider.getForwardComputationLauncher()
                .apply(new ForwardComputationContainer(inputMatrix, correctedNeuralNetworkModel));

        List<GradientDescentCorrection> gradientDescentCorrections = gradientDescentProvider.getBackwardComputationLauncher()
                .apply(new BackwardComputationContainer(provider, y, gradientDescentProvider.getFirstErrorComputationLauncher(), gradientDescentProvider.getErrorComputationLauncher()));

        return gradientDescentProvider.getGradientDescentCorrectionsLauncher()
                .apply(new GradientDescentCorrectionsContainer(correctedNeuralNetworkModel, gradientDescentCorrections, y.getColumns(), learningRate))
                .getCorrectedNeuralNetworkModel();
    }

}
