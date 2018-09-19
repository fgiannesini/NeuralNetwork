package com.fgiannesini.neuralnetwork.model;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import com.fgiannesini.neuralnetwork.initializer.Initializer;
import org.jblas.DoubleMatrix;

import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class ConvolutionLayer extends Layer {

    private List<DoubleMatrix> weightMatrices;
    private List<DoubleMatrix> biasMatrices;

    private int padding;
    private int stride;

    public ConvolutionLayer(ActivationFunctionType activationFunctionType, Initializer initializer, Integer filterSize, int padding, int stride, int channels) {
        super(activationFunctionType);
        this.padding = padding;
        this.stride = stride;
        weightMatrices = IntStream.range(0, channels)
                .mapToObj(i -> initializer.initDoubleMatrix(filterSize, filterSize))
                .collect(Collectors.toList());
        biasMatrices = IntStream.range(0, channels)
                .mapToObj(i -> initializer.initDoubleMatrix(1, 1))
                .collect(Collectors.toList());
    }
}
