package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.processprovider;

import com.fgiannesini.neuralnetwork.computer.data.BatchNormData;
import com.fgiannesini.neuralnetwork.computer.data.DataVisitor;
import com.fgiannesini.neuralnetwork.computer.data.LayerTypeData;
import com.fgiannesini.neuralnetwork.computer.data.WeightBiasData;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.layerdataprovider.GradientLayerProvider;
import org.jblas.DoubleMatrix;

public class GradientDescentSoftMaxRegressionVisitor implements DataVisitor {

    private final GradientLayerProvider provider;
    private LayerTypeData errorData;

    GradientDescentSoftMaxRegressionVisitor(GradientLayerProvider provider) {
        this.provider = provider;
    }

    @Override
    public void visit(WeightBiasData previousError) {
        WeightBiasData currentResult = (WeightBiasData) provider.getCurrentResult();
        DoubleMatrix error = computeError(previousError.getData(), currentResult.getData());
        errorData = new WeightBiasData(error);
    }

    @Override
    public void visit(BatchNormData previousError) {
        BatchNormData currentResult = (BatchNormData) provider.getCurrentResult();
        DoubleMatrix error = computeError(previousError.getData(), currentResult.getData());
        errorData = new BatchNormData(error, previousError.getMeanDeviationProvider());
    }

    private DoubleMatrix computeError(DoubleMatrix input, DoubleMatrix currentResult) {
        //dZ2 = A2-Y .* g2'(A2)
        return currentResult.sub(input)
                .muli(provider.getActivationFunction().derivate(currentResult));
    }

    LayerTypeData getErrorData() {
        return errorData;
    }
}
