package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.processprovider;

import com.fgiannesini.neuralnetwork.computer.BatchNormData;
import com.fgiannesini.neuralnetwork.computer.DataVisitor;
import com.fgiannesini.neuralnetwork.computer.LayerTypeData;
import com.fgiannesini.neuralnetwork.computer.WeightBiasData;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.layerdataprovider.GradientBatchNormLayerProvider;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.layerdataprovider.GradientLayerProvider;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.layerdataprovider.GradientWeightBiasLayerProvider;
import org.jblas.DoubleMatrix;

public class GradientDescentErrorComputationVisitor implements DataVisitor {
    private final GradientLayerProvider provider;
    private LayerTypeData errorData;

    GradientDescentErrorComputationVisitor(GradientLayerProvider provider) {
        this.provider = provider;
    }

    @Override
    public void visit(WeightBiasData previousError) {
        //dZ1 = W2t * dZ2 .* g1'(A1)
        WeightBiasData currentResult = (WeightBiasData) provider.getCurrentResult();
        DoubleMatrix error = ((GradientWeightBiasLayerProvider) provider).getPreviousWeightMatrix().transpose()
                .mmul(previousError.getData())
                .muli(provider.getActivationFunction().derivate(currentResult.getData()));
        errorData = new WeightBiasData(error);
    }

    @Override
    public void visit(BatchNormData previousError) {
        //dZ1 = W2t * dZ2 .* g1'(A1)
        BatchNormData currentResult = (BatchNormData) provider.getCurrentResult();
        DoubleMatrix error = ((GradientBatchNormLayerProvider) provider).getPreviousWeightMatrix().transpose()
                .mmul(previousError.getInput())
                .muli(provider.getActivationFunction().derivate(currentResult.getInput()));
        errorData = new BatchNormData(error, previousError.getMeanDeviationProvider());
    }

    LayerTypeData getErrorData() {
        return errorData;
    }
}
