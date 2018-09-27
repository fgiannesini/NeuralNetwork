package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.processprovider;

import com.fgiannesini.neuralnetwork.computer.BatchNormData;
import com.fgiannesini.neuralnetwork.computer.DataVisitor;
import com.fgiannesini.neuralnetwork.computer.LayerTypeData;
import com.fgiannesini.neuralnetwork.computer.WeightBiasData;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.layerdataprovider.GradientLayerProvider;
import org.jblas.DoubleMatrix;

public class GradientDescentLinearRegressionVisitor implements DataVisitor {

    private final GradientLayerProvider provider;
    private LayerTypeData errorData;

    GradientDescentLinearRegressionVisitor(GradientLayerProvider provider) {
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
        //dZ2 = (A2 - Y) .* g2'(A2)
        return provider.getCurrentResult()
                .sub(input)
                .muli(provider.getActivationFunction().derivate(provider.getCurrentResult()));
    }

    LayerTypeData getErrorData() {
        return errorData;
    }
}
