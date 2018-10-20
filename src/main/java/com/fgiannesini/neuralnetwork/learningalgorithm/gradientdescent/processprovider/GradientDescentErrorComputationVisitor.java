package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.processprovider;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionApplier;
import com.fgiannesini.neuralnetwork.computer.DataFunctionApplier;
import com.fgiannesini.neuralnetwork.computer.data.*;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.layerdataprovider.GradientLayerProvider;
import org.jblas.DoubleMatrix;

import java.util.ArrayList;
import java.util.List;

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

    @Override
    public void visit(ConvolutionData error) {
        ConvolutionData currentResult = (ConvolutionData) provider.getCurrentResult();
        List<DoubleMatrix> result = visitMatrixList(currentResult.getDatas(), error.getDatas());
        errorData = new ConvolutionData(result, error.getChannelCount());
    }

    @Override
    public void visit(AveragePoolingData error) {
        AveragePoolingData currentResult = (AveragePoolingData) provider.getCurrentResult();
        List<DoubleMatrix> result = visitMatrixList(currentResult.getDatas(), error.getDatas());
        errorData = new AveragePoolingData(result, error.getChannelCount());
    }

    @Override
    public void visit(MaxPoolingData error) {
        MaxPoolingData currentResult = (MaxPoolingData) provider.getCurrentResult();
        List<DoubleMatrix> result = visitMatrixList(currentResult.getDatas(), error.getDatas());
        errorData = new MaxPoolingData(result, error.getMaxRowIndexes(), error.getMaxColumnIndexes(), error.getChannelCount());
    }

    private List<DoubleMatrix> visitMatrixList(List<DoubleMatrix> currentResultList, List<DoubleMatrix> errorDatas) {
        ActivationFunctionApplier activationFunction = provider.getLayer().getActivationFunctionType().getActivationFunction();
        List<DoubleMatrix> result = new ArrayList<>();
        for (int matrixIndex = 0; matrixIndex < currentResultList.size(); matrixIndex++) {
            result.add(activationFunction.derivate(currentResultList.get(matrixIndex)).muli(errorDatas.get(matrixIndex)));
        }
        return result;
    }

    LayerTypeData getErrorData() {
        return errorData;
    }
}
