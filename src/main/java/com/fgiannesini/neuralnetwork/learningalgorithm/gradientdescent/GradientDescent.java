package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent;

import com.fgiannesini.neuralnetwork.learningalgorithm.LearningAlgorithm;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import org.jblas.DoubleMatrix;

import java.util.List;
import java.util.function.Function;

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
        GradientLayerProvider provider = gradientDescentProvider.getForwardComputationLauncher().apply(new ForwardComputationContainer(inputMatrix, correctedNeuralNetworkModel));
        List<GradientDescentCorrection> gradientDescentCorrections = gradientDescentProvider.getBackwardComputationLauncher()
                .apply(new BackwardComputationContainer(provider, y, gradientDescentProvider.getFirstErrorComputationLauncher(), gradientDescentProvider.getErrorComputationLauncher()));
        return applyGradientDescentCorrections(gradientDescentCorrections, y.getColumns());
    }

    protected NeuralNetworkModel applyGradientDescentCorrections(List<GradientDescentCorrection> gradientDescentCorrections, int inputCount) {
        GradientDescentCorrectionsContainer gradientDescentCorrectionsContainer = new GradientDescentCorrectionsContainer(correctedNeuralNetworkModel, gradientDescentCorrections, inputCount, learningRate);
        Function<GradientDescentCorrectionsContainer, NeuralNetworkModel> gradientDescentCorrectionsLauncher = gradientDescentProvider.getGradientDescentCorrectionsLauncher();
        return gradientDescentCorrectionsLauncher.apply(gradientDescentCorrectionsContainer);
    }

}
