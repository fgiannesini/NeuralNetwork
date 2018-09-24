package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.processprovider;

import com.fgiannesini.neuralnetwork.computer.OutputComputerBuilder;
import com.fgiannesini.neuralnetwork.computer.intermediateoutputcomputer.IIntermediateOutputComputer;
import com.fgiannesini.neuralnetwork.computer.intermediateoutputcomputer.IntermediateOutputResult;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.container.*;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.layerdataprovider.GradientBatchNormLayerProvider;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.layerdataprovider.GradientLayerProvider;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.layerdataprovider.GradientLayerProviderBuilder;
import com.fgiannesini.neuralnetwork.model.Layer;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import org.jblas.DoubleMatrix;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.function.Function;

public class GradientDescentBatchNormProcessProvider implements IGradientDescentProcessProvider {

    @Override
    public IGradientDescentProcessProvider getPreviousProcessProvider() {
        throw new RuntimeException("Should not be called");
    }

    @Override
    public Function<GradientDescentCorrectionsContainer, GradientDescentCorrectionsContainer> getGradientDescentCorrectionsLauncher() {
        return container -> {
            NeuralNetworkModel correctedNeuralNetworkModel = container.getCorrectedNeuralNetworkModel();
            List<Layer> layers = correctedNeuralNetworkModel.getLayers();
            for (int layerIndex = 0; layerIndex < layers.size(); layerIndex++) {
                GradientDescentCorrection gradientDescentCorrection = container.getGradientDescentCorrections().get(layerIndex);
                Layer layer = layers.get(layerIndex);
                List<DoubleMatrix> parametersMatrices = layer.getParametersMatrix();
                for (int parameterIndex = 0; parameterIndex < parametersMatrices.size(); parameterIndex++) {
                    parametersMatrices.get(parameterIndex).subi(gradientDescentCorrection.getCorrectionResults().get(parameterIndex).mul(container.getLearningRate()));
                }
            }
            return new GradientDescentCorrectionsContainer(correctedNeuralNetworkModel, container.getGradientDescentCorrections(), container.getInputCount(), container.getLearningRate());
        };
    }

    @Override
    public Function<BackwardComputationContainer, List<GradientDescentCorrection>> getBackwardComputationLauncher() {
        return container -> {
            List<GradientDescentCorrection> gradientDescentCorrections = new ArrayList<>();
            int inputCount = container.getY().getColumns();

            GradientBatchNormLayerProvider gradientLayerProvider = (GradientBatchNormLayerProvider) container.getProvider();
            DoubleMatrix dz = container.getFirstErrorComputationLauncher()
                    .apply(new ErrorComputationContainer(gradientLayerProvider, container.getY()))
                    .getPreviousError();
            BatchNormBackwardReturn batchNormBackwardReturn = getBatchNormBackwardReturn(inputCount, gradientLayerProvider, dz);

            gradientDescentCorrections.add(batchNormBackwardReturn.getCorrections());

            for (gradientLayerProvider.nextLayer(); gradientLayerProvider.hasNextLayer(); gradientLayerProvider.nextLayer()) {
                dz = container.getErrorComputationLauncher()
                        .apply(new ErrorComputationContainer(gradientLayerProvider, batchNormBackwardReturn.getNextError()))
                        .getPreviousError();
                batchNormBackwardReturn = getBatchNormBackwardReturn(inputCount, gradientLayerProvider, dz);
                gradientDescentCorrections.add(batchNormBackwardReturn.getCorrections());
            }

            Collections.reverse(gradientDescentCorrections);
            return gradientDescentCorrections;
        };

    }

    private BatchNormBackwardReturn getBatchNormBackwardReturn(int inputCount, GradientBatchNormLayerProvider gradientLayerProvider, DoubleMatrix dz) {
//        https://kevinzakka.github.io/2016/09/14/batch_normalization/
//        dxhat = dout * gamma
        DoubleMatrix dXhat = dz.mulColumnVector(gradientLayerProvider.getGammaMatrix());
        DoubleMatrix beforeActivationResult = gradientLayerProvider.getBeforeNormalisationCurrentResult();
//        dx = (1. / N) * inv_var * (N * dxhat - np.sum(dxhat, axis = 0)
//                - x_hat * np.sum(dxhat * x_hat, axis = 0))
        DoubleMatrix p1 = dXhat.mul(inputCount);
        DoubleMatrix p2 = dXhat.rowSums();
        DoubleMatrix p3 = beforeActivationResult.mulColumnVector(dXhat.mul(beforeActivationResult).rowSums());
        DoubleMatrix dx = p1.subiColumnVector(p2).subi(p3).divi(inputCount).diviColumnVector(gradientLayerProvider.getStandardDeviation());

//        dbeta = np.sum(dout, axis = 0)
        DoubleMatrix dBeta = dz.rowMeans();
//        dgamma = np.sum(x_hat * dout, axis = 0)
        DoubleMatrix dGamma = dz.mul(beforeActivationResult).rowMeans();

        DoubleMatrix weightCorrection = computeWeightCorrection(gradientLayerProvider.getPreviousResult(), dx, inputCount);
        return new BatchNormBackwardReturn(weightCorrection, dGamma, dBeta, dx);
    }

    private DoubleMatrix computeWeightCorrection(DoubleMatrix previousaLayerResult, DoubleMatrix dz, int inputCount) {
        //dW1 = dZ1 * A0t ./m
        return dz
                .mmul(previousaLayerResult.transpose())
                .divi(inputCount);
    }

    @Override
    public Function<ErrorComputationContainer, ErrorComputationContainer> getErrorComputationLauncher() {
        return container -> {
            //dZ1 = W2t * dZ2 .* g1'(A1)
            GradientBatchNormLayerProvider provider = (GradientBatchNormLayerProvider) container.getProvider();
            DoubleMatrix error = provider.getPreviousWeightMatrix().transpose()
                    .mmul(container.getPreviousError())
                    .muli(provider.getCurrentActivationFunction().derivate(provider.getCurrentResult()));
            return new ErrorComputationContainer(provider, error);
        };
    }

    @Override
    public Function<ErrorComputationContainer, ErrorComputationContainer> getFirstErrorComputationLauncher() {
        return container -> {
            throw new RuntimeException("Should use a regression type process provider");
        };
    }

    @Override
    public Function<ForwardComputationContainer, GradientLayerProvider> getForwardComputationLauncher() {
        return container -> {
            NeuralNetworkModel neuralNetworkModel = container.getNeuralNetworkModel();
            IIntermediateOutputComputer intermediateOutputComputer = OutputComputerBuilder.init()
                    .withModel(neuralNetworkModel)
                    .buildIntermediateOutputComputer();
            DoubleMatrix inputMatrix = container.getInput();
            List<IntermediateOutputResult> intermediateResults = intermediateOutputComputer.compute(inputMatrix);
            return GradientLayerProviderBuilder.init()
                    .withModel(neuralNetworkModel)
                    .withIntermediateResults(intermediateResults)
                    .build();
        };
    }

    @Override
    public Function<DataContainer, DataContainer> getDataProcessLauncher() {
        return Function.identity();
    }
}
