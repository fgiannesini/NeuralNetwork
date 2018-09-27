package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.processprovider;

import com.fgiannesini.neuralnetwork.computer.BatchNormData;
import com.fgiannesini.neuralnetwork.computer.DataVisitor;
import com.fgiannesini.neuralnetwork.computer.LayerTypeData;
import com.fgiannesini.neuralnetwork.computer.WeightBiasData;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.layerdataprovider.GradientLayerProvider;
import org.jblas.DoubleMatrix;

public class GradientDescentLogisticRegressionVisitor implements DataVisitor {

    private final GradientLayerProvider provider;
    private LayerTypeData errorData;

    GradientDescentLogisticRegressionVisitor(GradientLayerProvider provider) {
        this.provider = provider;
    }

    @Override
    public void visit(WeightBiasData previousError) {
        DoubleMatrix error = computeError(previousError.getInput());
        errorData = new WeightBiasData(error);
    }

    @Override
    public void visit(BatchNormData previousError) {
        DoubleMatrix error = computeError(previousError.getInput());
        errorData = new BatchNormData(error, previousError.getMeanDeviationProvider());
    }

    private DoubleMatrix computeError(DoubleMatrix input) {
        //dZ2 = (A2 - Y)/A2(1-A2)) .* g2'(A2)
        return provider.getCurrentResult().sub(input)
                .divi(provider.getCurrentResult())
                .divi(provider.getCurrentResult().neg().addi(1))
                .muli(provider.getActivationFunction().derivate(provider.getCurrentResult()));
    }

    LayerTypeData getErrorData() {
        return errorData;
    }
}
