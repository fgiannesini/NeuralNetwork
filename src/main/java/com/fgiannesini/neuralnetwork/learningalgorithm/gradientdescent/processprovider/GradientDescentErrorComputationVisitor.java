package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.processprovider;

import com.fgiannesini.neuralnetwork.computer.*;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.layerdataprovider.GradientLayerProvider;

public class GradientDescentErrorComputationVisitor implements DataVisitor {
    private final GradientLayerProvider provider;
    private LayerTypeData errorData;

    GradientDescentErrorComputationVisitor(GradientLayerProvider provider) {
        this.provider = provider;
    }

    @Override
    public void visit(WeightBiasData error) {
        WeightBiasData currentResult = (WeightBiasData) provider.getCurrentResult();
        DataFunctionApplier dataFunctionApplier = new DataFunctionApplier(matrix -> matrix.mul(provider.getLayer().getActivationFunctionType().getActivationFunction().derivate(currentResult.getData())));
        error.accept(dataFunctionApplier);
        errorData = dataFunctionApplier.getLayerTypeData();
    }

    @Override
    public void visit(BatchNormData error) {
        BatchNormData currentResult = (BatchNormData) provider.getCurrentResult();
        DataFunctionApplier dataFunctionApplier = new DataFunctionApplier(matrix -> matrix.mul(provider.getLayer().getActivationFunctionType().getActivationFunction().derivate(currentResult.getData())));
        error.accept(dataFunctionApplier);
        errorData = dataFunctionApplier.getLayerTypeData();
    }

    LayerTypeData getErrorData() {
        return errorData;
    }
}
