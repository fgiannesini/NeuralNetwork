package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent;

import com.fgiannesini.neuralnetwork.computer.BatchNormData;
import com.fgiannesini.neuralnetwork.computer.DataVisitor;
import com.fgiannesini.neuralnetwork.computer.LayerTypeData;
import com.fgiannesini.neuralnetwork.computer.WeightBiasData;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.container.GradientDescentCorrection;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.layerdataprovider.GradientBatchNormLayerProvider;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.layerdataprovider.GradientLayerProvider;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.processprovider.BatchNormBackwardReturn;
import com.fgiannesini.neuralnetwork.model.BatchNormLayer;
import com.fgiannesini.neuralnetwork.model.WeightBiasLayer;
import org.jblas.DoubleMatrix;

public class LayerTypeCorrectionsVisitor implements DataVisitor {
    private final GradientLayerProvider gradientLayerProvider;
    private LayerTypeData nextGradientLayerProvider;
    private GradientDescentCorrection correction;

    public LayerTypeCorrectionsVisitor(GradientLayerProvider gradientLayerProvider) {
        this.gradientLayerProvider = gradientLayerProvider;
    }

    @Override
    public void visit(WeightBiasData error) {
        DoubleMatrix errorInput = error.getData();
        int inputCount = errorInput.getColumns();
        WeightBiasData previousResult = (WeightBiasData) gradientLayerProvider.getPreviousResult();
        DoubleMatrix weightCorrection = computeWeightCorrection(previousResult.getData(), errorInput, inputCount);
        DoubleMatrix biasCorrection = computeBiasCorrection(errorInput, inputCount);
        correction = new GradientDescentCorrection(weightCorrection, biasCorrection);

        //dZ1 = W2t * dZ2
        WeightBiasLayer layer = (WeightBiasLayer) gradientLayerProvider.getLayer();
        DoubleMatrix nextError = layer.getWeightMatrix().transpose()
                .mmul(error.getData());
        nextGradientLayerProvider = new WeightBiasData(nextError);
    }

    private DoubleMatrix computeBiasCorrection(DoubleMatrix dz, int inputCount) {
        //dB = sum(dZ) ./ m
        return dz.rowSums()
                .divi(inputCount);
    }

    private DoubleMatrix computeWeightCorrection(DoubleMatrix previousLayerResult, DoubleMatrix dz, int inputCount) {
        //dW1 = dZ1 * A0t ./m
        return dz
                .mmul(previousLayerResult.transpose())
                .divi(inputCount);
    }

    @Override
    public void visit(BatchNormData error) {
        DoubleMatrix errorInput = error.getData();
        int inputCount = errorInput.getColumns();
        BatchNormBackwardReturn batchNormBackwardReturn = getBatchNormBackwardReturn(inputCount, (GradientBatchNormLayerProvider) gradientLayerProvider, errorInput);
        correction = batchNormBackwardReturn.getCorrections();

        //dZ1 = W2t * dZ2
        BatchNormLayer layer = (BatchNormLayer) gradientLayerProvider.getLayer();
        DoubleMatrix nextError = layer.getWeightMatrix().transpose()
                .mmul(batchNormBackwardReturn.getNextError());

        nextGradientLayerProvider = new BatchNormData(nextError, error.getMeanDeviationProvider());
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

        BatchNormData previousResult = (BatchNormData) gradientLayerProvider.getPreviousResult();
        DoubleMatrix weightCorrection = computeWeightCorrection(previousResult.getData(), dx, inputCount);
        return new BatchNormBackwardReturn(weightCorrection, dGamma, dBeta, dx);
    }

    public GradientDescentCorrection getCorrection() {
        return correction;
    }

    public LayerTypeData getNextGradientLayerProvider() {
        return nextGradientLayerProvider;
    }
}
