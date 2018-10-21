package com.fgiannesini.neuralnetwork.computer.intermediateoutputcomputer;

import com.fgiannesini.neuralnetwork.computer.data.*;
import org.jblas.DoubleMatrix;

import java.util.List;
import java.util.stream.Collectors;

public class InputDropOutRegularizationVisitor implements DataVisitor {
    private DoubleMatrix regularizationMatrix;
    private LayerTypeData regularizedData;

    public InputDropOutRegularizationVisitor(DoubleMatrix regularizationMatrix) {
        this.regularizationMatrix = regularizationMatrix;
    }

    @Override
    public void visit(WeightBiasData data) {
        regularizedData = new WeightBiasData(data.getData().dup().muliColumnVector(regularizationMatrix));
    }

    @Override
    public void visit(BatchNormData data) {
        regularizedData = new BatchNormData(data.getData().dup().muliColumnVector(regularizationMatrix), data.getMeanDeviationProvider());
    }

    @Override
    public void visit(ConvolutionData convolutionData) {
        List<DoubleMatrix> regularizedMatrices = convolutionData.getDatas().stream()
                .map(matrix -> matrix.muli(regularizationMatrix))
                .collect(Collectors.toList());
        regularizedData = new ConvolutionData(regularizedMatrices, convolutionData.getChannelCount());
    }

    @Override
    public void visit(AveragePoolingData averagePoolingData) {
        regularizedData = averagePoolingData;
    }

    @Override
    public void visit(MaxPoolingData maxPoolingData) {
        regularizedData = maxPoolingData;
    }

    public LayerTypeData getRegularizedData() {
        return regularizedData;
    }
}
