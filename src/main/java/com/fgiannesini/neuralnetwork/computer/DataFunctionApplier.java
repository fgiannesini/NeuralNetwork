package com.fgiannesini.neuralnetwork.computer;

import org.jblas.DoubleMatrix;

import java.util.function.Function;
import java.util.stream.Collectors;

public class DataFunctionApplier implements DataVisitor {

    private final Function<DoubleMatrix, DoubleMatrix> dataApplier;
    private LayerTypeData layerTypeData;

    public DataFunctionApplier(Function<DoubleMatrix, DoubleMatrix> dataApplier) {
        this.dataApplier = dataApplier;
    }

    @Override
    public void visit(WeightBiasData data) {
        layerTypeData = new WeightBiasData(dataApplier.apply(data.getData()));
    }

    @Override
    public void visit(BatchNormData data) {
        layerTypeData = new BatchNormData(dataApplier.apply(data.getInput()), data.getMeanDeviationProvider());
    }

    @Override
    public void visit(ConvolutionData convolutionData) {
        layerTypeData = new ConvolutionData(convolutionData.getDatas().stream().map(dataApplier).collect(Collectors.toList()));
    }

    public LayerTypeData getLayerTypeData() {
        return layerTypeData;
    }
}
