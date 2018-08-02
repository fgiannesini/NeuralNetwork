package com.fgiannesini.neuralnetwork.learningalgorithm.regularization.l2;

import com.fgiannesini.neuralnetwork.learningalgorithm.LearningAlgorithm;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.*;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import org.jblas.DoubleMatrix;

import java.util.List;

public class GradientDescentWithL2Regularization implements LearningAlgorithm {
    private final double learningRate;
    private final IGradientDescentProcessProvider gradientDescentProcessProvider;
    private final NeuralNetworkModel originalNeuralNetworkModel;

    public GradientDescentWithL2Regularization(NeuralNetworkModel originalNeuralNetworkModel, double learningRate, double regularizationCoeff) {
        this.originalNeuralNetworkModel = originalNeuralNetworkModel.clone();
        this.learningRate = learningRate;
        this.gradientDescentProcessProvider = new GradientDescentWithL2RegularizationProcessProvider(regularizationCoeff, this.originalNeuralNetworkModel);
    }

    @Override
    public NeuralNetworkModel learn(DoubleMatrix inputMatrix, DoubleMatrix y) {
        GradientLayerProvider gradientLayerProvider = gradientDescentProcessProvider.getForwardComputationLauncher()
                .apply(new ForwardComputationContainer(inputMatrix, originalNeuralNetworkModel));

        List<GradientDescentCorrection> gradientDescentCorrections = gradientDescentProcessProvider.getBackwardComputationLauncher()
                .apply(new BackwardComputationContainer(gradientLayerProvider, y, gradientDescentProcessProvider.getFirstErrorComputationLauncher(), gradientDescentProcessProvider.getErrorComputationLauncher()));

        return gradientDescentProcessProvider.getGradientDescentCorrectionsLauncher()
                .apply(new GradientDescentCorrectionsContainer(originalNeuralNetworkModel, gradientDescentCorrections, y.getColumns(), learningRate))
                .getCorrectedNeuralNetworkModel();

    }
}
