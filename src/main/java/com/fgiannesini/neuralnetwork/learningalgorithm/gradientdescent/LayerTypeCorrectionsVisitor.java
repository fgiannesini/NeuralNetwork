package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent;

import com.fgiannesini.neuralnetwork.computer.BatchNormData;
import com.fgiannesini.neuralnetwork.computer.DataVisitor;
import com.fgiannesini.neuralnetwork.computer.LayerTypeData;
import com.fgiannesini.neuralnetwork.computer.WeightBiasData;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.container.GradientDescentCorrection;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.layerdataprovider.GradientLayerProvider;
import org.jblas.DoubleMatrix;

public class LayerTypeCorrectionsVisitor implements DataVisitor {
    private final LayerTypeData firstError;
    private final GradientLayerProvider gradientLayerProvider;
    private GradientDescentCorrection correction;

    public LayerTypeCorrectionsVisitor(LayerTypeData firstError, GradientLayerProvider gradientLayerProvider) {
        this.firstError = firstError;
        this.gradientLayerProvider = gradientLayerProvider;
    }

    @Override
    public void visit(WeightBiasData data) {
        DoubleMatrix weightCorrection = computeWeightCorrection(gradientLayerProvider.getPreviousResult(), firstError, inputCount);
        DoubleMatrix biasCorrection = computeBiasCorrection(firstError, inputCount);

        correction = new GradientDescentCorrection(weightCorrection, biasCorrection);
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
    public void visit(BatchNormData data) {

    }

    public GradientDescentCorrection getCorrection() {
        return correction;
    }
}
