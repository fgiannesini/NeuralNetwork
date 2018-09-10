package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.processprovider;

import com.fgiannesini.neuralnetwork.computer.OutputComputerBuilder;
import com.fgiannesini.neuralnetwork.computer.intermediateoutputcomputer.IIntermediateOutputComputer;
import com.fgiannesini.neuralnetwork.computer.intermediateoutputcomputer.IntermediateOutputResult;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.container.*;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.layerdataprovider.GradientLayerProvider;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.layerdataprovider.GradientLayerProviderBuilder;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import com.fgiannesini.neuralnetwork.model.WeightBiasLayer;
import org.jblas.DoubleMatrix;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.function.Function;

public class GradientDescentWeightBiasProcessProvider implements IGradientDescentProcessProvider<WeightBiasLayer> {

    @Override
    public Function<GradientDescentCorrectionsContainer<WeightBiasLayer>, GradientDescentCorrectionsContainer<WeightBiasLayer>> getGradientDescentCorrectionsLauncher() {
        return container -> {
            NeuralNetworkModel<WeightBiasLayer> correctedNeuralNetworkModel = container.getCorrectedNeuralNetworkModel();
            List<WeightBiasLayer> layers = correctedNeuralNetworkModel.getLayers();
            for (int layerIndex = 0; layerIndex < layers.size(); layerIndex++) {
                GradientDescentCorrection gradientDescentCorrection = container.getGradientDescentCorrections().get(layerIndex);
                List<DoubleMatrix> parametersMatrices = layers.get(layerIndex).getParametersMatrix();
                for (int parameterIndex = 0; parameterIndex < parametersMatrices.size(); parameterIndex++) {
                    parametersMatrices.get(parameterIndex).subi(gradientDescentCorrection.getCorrectionResults().get(parameterIndex).mul(container.getLearningRate()));
                }
            }
            return new GradientDescentCorrectionsContainer<>(correctedNeuralNetworkModel, container.getGradientDescentCorrections(), container.getInputCount(), container.getLearningRate());
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
            DoubleMatrix delta = container.getProvider().getPreviousWeightMatrix().transpose()
                    .mmul(container.getPreviousError());
            DoubleMatrix error = container.getProvider().getCurrentActivationFunction().derivate(container.getProvider().getCurrentResult(), delta);
            return new ErrorComputationContainer(container.getProvider(), error);
        };
    }

    @Override
    public Function<ErrorComputationContainer, ErrorComputationContainer> getFirstErrorComputationLauncher() {
        return container -> {
            throw new RuntimeException("Should use a regression type process provider");
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
    public Function<ForwardComputationContainer<WeightBiasLayer>, GradientLayerProvider<WeightBiasLayer>> getForwardComputationLauncher() {
        return container -> {
            IIntermediateOutputComputer intermediateOutputComputer = OutputComputerBuilder.init()
                    .withModel(container.getNeuralNetworkModel())
                    .buildIntermediateOutputComputer();
            List<IntermediateOutputResult> intermediateResults = intermediateOutputComputer.compute(container.getInputMatrix());
            return GradientLayerProviderBuilder.init()
                    .withModel(container.getNeuralNetworkModel())
                    .withIntermediateResults(intermediateResults)
                    .build();
        };
    }

    @Override
    public Function<DataContainer, DataContainer> getDataProcessLauncher() {
        return Function.identity();
    }
}
