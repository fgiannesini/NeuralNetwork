package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import com.fgiannesini.neuralnetwork.initializer.InitializerType;
import com.fgiannesini.neuralnetwork.model.Layer;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;

public class GradientDescentWithRmsStopProcessProvider implements IGradientDescentProcessProvider {
    private final List<Layer> rmsStopLayers;
    private final IGradientDescentProcessProvider processProvider;
    private final Double rmsStopCoeff;
    private final Double epsilon;

    public GradientDescentWithRmsStopProcessProvider(Double rmsStopCoeff) {
        this.rmsStopCoeff = rmsStopCoeff;
        this.epsilon = Math.pow(10, -8);
        processProvider = new GradientDescentProcessProvider();
        rmsStopLayers = new ArrayList<>();
    }

    @Override
    public Function<GradientDescentCorrectionsContainer, GradientDescentCorrectionsContainer> getGradientDescentCorrectionsLauncher() {
        return container -> {
            NeuralNetworkModel correctedNeuralNetworkModel = container.getCorrectedNeuralNetworkModel();
            List<Layer> layers = correctedNeuralNetworkModel.getLayers();
            if (rmsStopLayers.isEmpty()) {
                rmsStopLayers.addAll(initRmsStopLayers(layers));
            }
            for (int layerIndex = 0; layerIndex < layers.size(); layerIndex++) {
                GradientDescentCorrection gradientDescentCorrection = container.getGradientDescentCorrections().get(layerIndex);
                Layer layer = layers.get(layerIndex);
                Layer rmsStopLayer = rmsStopLayers.get(layerIndex);

                //Sdw = c * Sdw + (1 - c)*dW²
                rmsStopLayer.setWeightMatrix(rmsStopLayer.getWeightMatrix().muli(rmsStopCoeff).addi(MatrixFunctions.pow(gradientDescentCorrection.getWeightCorrectionResults(), 2d).muli(1d - rmsStopCoeff)));
                //W = W - a * dW/(sqrt(Sdw) + e)
                DoubleMatrix weightCorrection = gradientDescentCorrection.getWeightCorrectionResults().div(MatrixFunctions.sqrt(rmsStopLayer.getWeightMatrix()).addi(epsilon)).muli(container.getLearningRate());
                layer.getWeightMatrix().subi(weightCorrection);

                //Sdb = c *Sdb + (1 - c)*dB²
                rmsStopLayer.setBiasMatrix(rmsStopLayer.getBiasMatrix().muli(rmsStopCoeff).addi(MatrixFunctions.pow(gradientDescentCorrection.getBiasCorrectionResults(), 2d).muli(1d - rmsStopCoeff)));
                //B = B - a * dB/(sqrt(Sdb) + e)
                DoubleMatrix biasCorrection = gradientDescentCorrection.getBiasCorrectionResults().div(MatrixFunctions.sqrt(rmsStopLayer.getBiasMatrix()).addi(epsilon)).muli(container.getLearningRate());
                layer.getBiasMatrix().subi(biasCorrection);
            }
            return new GradientDescentCorrectionsContainer(correctedNeuralNetworkModel, container.getGradientDescentCorrections(), container.getInputCount(), container.getLearningRate());
        };
    }

    private List<Layer> initRmsStopLayers(List<Layer> layers) {
        return layers.stream()
                .map(layer -> new Layer(layer.getInputLayerSize(), layer.getOutputLayerSize(), InitializerType.ZEROS.getInitializer(), ActivationFunctionType.NONE))
                .collect(Collectors.toList());
    }


    @Override
    public Function<BackwardComputationContainer, List<GradientDescentCorrection>> getBackwardComputationLauncher() {
        return processProvider.getBackwardComputationLauncher();
    }

    @Override
    public Function<ErrorComputationContainer, ErrorComputationContainer> getErrorComputationLauncher() {
        return processProvider.getErrorComputationLauncher();
    }

    @Override
    public Function<ErrorComputationContainer, ErrorComputationContainer> getFirstErrorComputationLauncher() {
        return processProvider.getFirstErrorComputationLauncher();
    }

    @Override
    public Function<ForwardComputationContainer, GradientLayerProvider> getForwardComputationLauncher() {
        return processProvider.getForwardComputationLauncher();
    }

    @Override
    public Function<DataContainer, DataContainer> getDataProcessLauncher() {
        return processProvider.getDataProcessLauncher();
    }
}
