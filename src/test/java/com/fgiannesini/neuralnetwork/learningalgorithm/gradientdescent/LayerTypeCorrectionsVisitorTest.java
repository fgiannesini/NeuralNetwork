package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import com.fgiannesini.neuralnetwork.assertions.DoubleMatrixAssertions;
import com.fgiannesini.neuralnetwork.computer.data.ConvolutionData;
import com.fgiannesini.neuralnetwork.initializer.InitializerType;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.layerdataprovider.GradientConvolutionLayerProvider;
import com.fgiannesini.neuralnetwork.model.ConvolutionNeuralNetworkModelBuilder;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import org.jblas.DoubleMatrix;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import java.util.Collections;
import java.util.stream.IntStream;

class LayerTypeCorrectionsVisitorTest {

    @Nested
    class ConvolutionLayer {

        @Test
        void nominal() {
            NeuralNetworkModel neuralNetworkModel = ConvolutionNeuralNetworkModelBuilder.init()
                    .useInitializer(InitializerType.ONES)
                    .input(5, 5, 1)
                    .addConvolutionLayer(3, 0, 1, 1, ActivationFunctionType.NONE)
                    .buildConvolutionNetworkModel();

            neuralNetworkModel.getLayers().get(0).getParametersMatrix().get(0).muli(new DoubleMatrix(3, 3, IntStream.range(1, 10).asDoubleStream().map(d -> d / 100).toArray()));
            ConvolutionData previousResults = new ConvolutionData(Collections.singletonList(new DoubleMatrix(5, 5, IntStream.range(1, 26).asDoubleStream().toArray())));
            ConvolutionData results = new ConvolutionData(Collections.singletonList(new DoubleMatrix(3, 3, IntStream.range(1, 10).asDoubleStream().toArray())));
            GradientConvolutionLayerProvider layerProvider = new GradientConvolutionLayerProvider(neuralNetworkModel.getLayers().get(0), results, previousResults, 1);

            ConvolutionData error = new ConvolutionData(Collections.singletonList(new DoubleMatrix(3, 3, IntStream.range(1, 10).asDoubleStream().map(d -> d / 10).toArray())));
            LayerTypeCorrectionsVisitor dataVisitor = new LayerTypeCorrectionsVisitor(layerProvider);
            error.accept(dataVisitor);

            DoubleMatrix weightResults = dataVisitor.getCorrection().getCorrectionResults().get(0);
            DoubleMatrix biasResults = dataVisitor.getCorrection().getCorrectionResults().get(1);

            DoubleMatrix errorDatas = ((ConvolutionData) dataVisitor.getNextGradientLayerProvider()).getDatas().get(0);

            DoubleMatrixAssertions.assertMatrices(new DoubleMatrix(3, 3, 41.1, 45.6, 50.1, 63.6, 68.1, 72.6, 86.1, 90.6, 95.1), weightResults);
            DoubleMatrixAssertions.assertMatrices(new DoubleMatrix(1, 1, 4.5), biasResults);
            DoubleMatrixAssertions.assertMatrices(new DoubleMatrix(5, 5, 0.001, 0.004, 0.01, 0.012, 0.009, 0.008, 0.026, 0.056, 0.054, 0.036, 0.03, 0.084, 0.165, 0.144, 0.09, 0.056, 0.134, 0.236, 0.186, 0.108, 0.049, 0.112, 0.19, 0.144, 0.081), errorDatas);
        }

    }

    @Nested
    class MaxPoolingLayer {

    }

    @Nested
    class AveragePoolingLayer {

    }

}