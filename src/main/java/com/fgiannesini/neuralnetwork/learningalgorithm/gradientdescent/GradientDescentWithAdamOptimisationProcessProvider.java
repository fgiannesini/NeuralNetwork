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

public class GradientDescentWithAdamOptimisationProcessProvider implements IGradientDescentProcessProvider {
    private final Double momentumCoeff;
    private final Double rmsStopCoeff;
    private final GradientDescentOnLinearRegressionProcessProvider processProvider;
    private final List<Layer> momentumLayers;
    private final List<Layer> rmsStopLayers;
    private final Double epsilon;

    public GradientDescentWithAdamOptimisationProcessProvider(Double momentumCoeff, Double rmsStopCoeff) {
        this.momentumCoeff = momentumCoeff;
        this.rmsStopCoeff = rmsStopCoeff;
        this.processProvider = new GradientDescentOnLinearRegressionProcessProvider();
        momentumLayers = new ArrayList<>();
        rmsStopLayers = new ArrayList<>();
        this.epsilon = Math.pow(10, -8);
    }

    @Override
    public Function<GradientDescentCorrectionsContainer, GradientDescentCorrectionsContainer> getGradientDescentCorrectionsLauncher() {
        return container -> {
            NeuralNetworkModel correctedNeuralNetworkModel = container.getCorrectedNeuralNetworkModel();
            List<Layer> layers = correctedNeuralNetworkModel.getLayers();
            if (momentumLayers.isEmpty()) {
                momentumLayers.addAll(initLayers(layers));
            }
            if (rmsStopLayers.isEmpty()) {
                rmsStopLayers.addAll(initLayers(layers));
            }
            for (int layerIndex = 0; layerIndex < layers.size(); layerIndex++) {
                GradientDescentCorrection gradientDescentCorrection = container.getGradientDescentCorrections().get(layerIndex);
                Layer layer = layers.get(layerIndex);
                Layer momentumLayer = momentumLayers.get(layerIndex);
                Layer rmsStopLayer = rmsStopLayers.get(layerIndex);

                //Vdw = m*Vdw + (1 - m)*dW
                momentumLayer.setWeightMatrix(momentumLayer.getWeightMatrix().muli(momentumCoeff).addi(gradientDescentCorrection.getWeightCorrectionResults().mul(1d - momentumCoeff)));
                //Sdw = c * Sdw + (1 - c)*dW²
                rmsStopLayer.setWeightMatrix(rmsStopLayer.getWeightMatrix().muli(rmsStopCoeff).addi(MatrixFunctions.pow(gradientDescentCorrection.getWeightCorrectionResults(), 2d).muli(1d - rmsStopCoeff)));
                //W = W - a * Vdw/(1-m)/(sqrt(Sdw/(1-c)) + e)
                DoubleMatrix weightCorrection = momentumLayer.getWeightMatrix().div(1d - momentumCoeff).divi(MatrixFunctions.sqrt(rmsStopLayer.getWeightMatrix().div(1d - rmsStopCoeff)).addi(epsilon)).muli(container.getLearningRate());
                layer.getWeightMatrix().subi(weightCorrection);

                //Vdb = m*Vdb + (1 - m)*dB
                momentumLayer.setBiasMatrix(momentumLayer.getBiasMatrix().muli(momentumCoeff).addi(gradientDescentCorrection.getBiasCorrectionResults().mul(1d - momentumCoeff)));
                //Sdb = c *Sdb + (1 - c)*dB²
                rmsStopLayer.setBiasMatrix(rmsStopLayer.getBiasMatrix().muli(rmsStopCoeff).addi(MatrixFunctions.pow(gradientDescentCorrection.getBiasCorrectionResults(), 2d).muli(1d - rmsStopCoeff)));
                //B = B - a * Vdb/(1-m)/(sqrt(Sdb/(1-c)) + e)
                DoubleMatrix biasCorrection = momentumLayer.getBiasMatrix().div(1d - momentumCoeff).divi(MatrixFunctions.sqrt(rmsStopLayer.getBiasMatrix().div(1d - rmsStopCoeff)).addi(epsilon)).muli(container.getLearningRate());
                layer.getBiasMatrix().subi(biasCorrection);
            }
            return new GradientDescentCorrectionsContainer(correctedNeuralNetworkModel, container.getGradientDescentCorrections(), container.getInputCount(), container.getLearningRate());
        };
    }

    private List<Layer> initLayers(List<Layer> layers) {
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
