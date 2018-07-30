package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent;

import com.fgiannesini.neuralnetwork.computer.OutputComputerBuilder;
import com.fgiannesini.neuralnetwork.computer.intermediateoutputcomputer.IIntermediateOutputComputer;
import com.fgiannesini.neuralnetwork.model.Layer;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import org.jblas.DoubleMatrix;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.function.Function;

public class GradientDescentProcessProvider implements IGradientDescentProcessProvider {

    @Override
    public Function<GradientDescentCorrectionsContainer, NeuralNetworkModel> getGradientDescentCorrectionsLauncher() {
        return container -> {
            NeuralNetworkModel correctedNeuralNetworkModel = container.getCorrectedNeuralNetworkModel();
            List<Layer> layers = correctedNeuralNetworkModel.getLayers();
            for (int layerIndex = 0; layerIndex < layers.size(); layerIndex++) {
                GradientDescentCorrection gradientDescentCorrection = container.getGradientDescentCorrections().get(layerIndex);
                Layer layer = layers.get(layerIndex);
                layer.getWeightMatrix().subi(gradientDescentCorrection.getWeightCorrectionResults().mul(container.getLearningRate()));
                layer.getBiasMatrix().subi(gradientDescentCorrection.getBiasCorrectionResults().mul(container.getLearningRate()));
            }
            return correctedNeuralNetworkModel;
        };
    }

    @Override
    public Function<BackwardComputationContainer, List<GradientDescentCorrection>> getBackwardComputationLauncher() {
        return container -> {
            List<GradientDescentCorrection> gradientDescentCorrections = new ArrayList<>();
            int inputCount = container.getY().getColumns();

            GradientLayerProvider gradientLayerProvider = container.getProvider();
            DoubleMatrix dz = container.getFirstErrorComputationLauncher()
                    .apply(new ErrorComputationContainer(gradientLayerProvider, container.getY()))
                    .getPreviousError();
            DoubleMatrix weightCorrection = computeWeightCorrection(gradientLayerProvider.getPreviousResult(), dz, inputCount);
            DoubleMatrix biasCorrection = computeBiasCorrection(dz, inputCount);

            gradientDescentCorrections.add(new GradientDescentCorrection(weightCorrection, biasCorrection));

            for (gradientLayerProvider.nextLayer(); gradientLayerProvider.hasNextLayer(); gradientLayerProvider.nextLayer()) {
                dz = container.getErrorComputationLauncher()
                        .apply(new ErrorComputationContainer(gradientLayerProvider, dz))
                        .getPreviousError();
                weightCorrection = computeWeightCorrection(gradientLayerProvider.getPreviousResult(), dz, inputCount);
                biasCorrection = computeBiasCorrection(dz, inputCount);
                gradientDescentCorrections.add(new GradientDescentCorrection(weightCorrection, biasCorrection));
            }

            Collections.reverse(gradientDescentCorrections);

            return gradientDescentCorrections;
        };
    }

    @Override
    public Function<ErrorComputationContainer, ErrorComputationContainer> getErrorComputationLauncher() {
        return container -> {
            //dZ1 = W2t * dZ2 .* g1'(A1)
            DoubleMatrix error = container.getProvider().getPreviousWeightMatrix().transpose()
                    .mmul(container.getPreviousError())
                    .muli(container.getProvider().getCurrentActivationFunction().derivate(container.getProvider().getCurrentResult()));
            return new ErrorComputationContainer(container.getProvider(), error);
        };
    }

    @Override
    public Function<ErrorComputationContainer, ErrorComputationContainer> getFirstErrorComputationLauncher() {
        return container -> {
            //dZ2 = (A2 - Y) .* g2'(A2)
            DoubleMatrix error = container.getProvider().getCurrentResult()
                    .sub(container.getPreviousError())
                    .muli(container.getProvider().getCurrentActivationFunction().derivate(container.getProvider().getCurrentResult()));
            return new ErrorComputationContainer(container.getProvider(), error);
        };
    }

    private DoubleMatrix computeBiasCorrection(DoubleMatrix dz, int inputCount) {
        //dB = sum(dZ) ./ m
        return dz.rowSums()
                .divi(inputCount);
    }

    private DoubleMatrix computeWeightCorrection(DoubleMatrix previousaLayerResult, DoubleMatrix dz, int inputCount) {
        //dW1 = dZ1 * A0t ./m
        return dz
                .mmul(previousaLayerResult.transpose())
                .divi(inputCount);
    }

    @Override
    public Function<ForwardComputationContainer, GradientLayerProvider> getForwardComputationLauncher() {
        return container -> {
            List<Layer> layers = container.getNeuralNetworkModel().getLayers();
            IIntermediateOutputComputer intermediateOutputComputer = OutputComputerBuilder.init()
                    .withModel(container.getNeuralNetworkModel())
                    .buildIntermediateOutputComputer();
            List<DoubleMatrix> intermediateResults = intermediateOutputComputer.compute(container.getInputMatrix());
            return new GradientLayerProvider(layers, intermediateResults);
        };
    }
}
