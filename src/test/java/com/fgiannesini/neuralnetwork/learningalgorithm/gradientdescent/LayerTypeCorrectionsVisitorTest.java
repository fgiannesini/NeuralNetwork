package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import com.fgiannesini.neuralnetwork.assertions.DoubleMatrixAssertions;
import com.fgiannesini.neuralnetwork.computer.data.AveragePoolingData;
import com.fgiannesini.neuralnetwork.computer.data.ConvolutionData;
import com.fgiannesini.neuralnetwork.computer.data.MaxPoolingData;
import com.fgiannesini.neuralnetwork.initializer.InitializerType;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.layerdataprovider.GradientConvolutionLayerProvider;
import com.fgiannesini.neuralnetwork.model.ConvolutionNeuralNetworkModelBuilder;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import org.jblas.DoubleMatrix;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.function.DoubleUnaryOperator;
import java.util.stream.IntStream;

class LayerTypeCorrectionsVisitorTest {

    private DoubleMatrix createMatrix(int size, DoubleUnaryOperator modifier) {
        return new DoubleMatrix(size, size, IntStream.range(1, size * size + 1).asDoubleStream().map(modifier).toArray());
    }

    @Nested
    class ConvolutionLayer {

        @Test
        void one_layer_no_padding_no_striding_one_channel() {
            NeuralNetworkModel neuralNetworkModel = ConvolutionNeuralNetworkModelBuilder.init()
                    .useInitializer(InitializerType.ONES)
                    .input(5, 5, 1)
                    .addConvolutionLayer(3, 0, 1, 1, ActivationFunctionType.NONE)
                    .buildConvolutionNetworkModel();

            neuralNetworkModel.getLayers().get(0).getParametersMatrix().get(0).muli(createMatrix(3, d -> d / 100));
            ConvolutionData previousResults = new ConvolutionData(Collections.singletonList(createMatrix(5, DoubleUnaryOperator.identity())), 1);
            ConvolutionData results = new ConvolutionData(Collections.singletonList(createMatrix(3, DoubleUnaryOperator.identity())), 1);
            GradientConvolutionLayerProvider layerProvider = new GradientConvolutionLayerProvider(neuralNetworkModel.getLayers().get(0), results, previousResults, 1);

            ConvolutionData error = new ConvolutionData(Collections.singletonList(createMatrix(3, d -> d / 10)), 1);
            LayerTypeCorrectionsVisitor dataVisitor = new LayerTypeCorrectionsVisitor(layerProvider);
            error.accept(dataVisitor);

            DoubleMatrix weightResults = dataVisitor.getCorrection().getCorrectionResults().get(0);
            DoubleMatrix biasResults = dataVisitor.getCorrection().getCorrectionResults().get(1);

            DoubleMatrix errorDatas = ((ConvolutionData) dataVisitor.getNextGradientLayerProvider()).getDatas().get(0);

            DoubleMatrixAssertions.assertMatrices(new DoubleMatrix(3, 3, 41.1, 45.6, 50.1, 63.6, 68.1, 72.6, 86.1, 90.6, 95.1), weightResults);
            DoubleMatrixAssertions.assertMatrices(new DoubleMatrix(1, 1, 4.5), biasResults);
            DoubleMatrixAssertions.assertMatrices(new DoubleMatrix(5, 5, 0.001, 0.004, 0.01, 0.012, 0.009, 0.008, 0.026, 0.056, 0.054, 0.036, 0.03, 0.084, 0.165, 0.144, 0.09, 0.056, 0.134, 0.236, 0.186, 0.108, 0.049, 0.112, 0.19, 0.144, 0.081), errorDatas);
        }

        @Test
        void one_layer_with_padding_no_striding_one_channel() {
            NeuralNetworkModel neuralNetworkModel = ConvolutionNeuralNetworkModelBuilder.init()
                    .useInitializer(InitializerType.ONES)
                    .input(4, 4, 1)
                    .addConvolutionLayer(3, 1, 1, 1, ActivationFunctionType.NONE)
                    .buildConvolutionNetworkModel();

            neuralNetworkModel.getLayers().get(0).getParametersMatrix().get(0).muli(createMatrix(3, d -> d / 100));
            ConvolutionData previousResults = new ConvolutionData(Collections.singletonList(createMatrix(4, DoubleUnaryOperator.identity())), 1);
            ConvolutionData results = new ConvolutionData(Collections.singletonList(createMatrix(4, d -> d / 2)), 1);
            GradientConvolutionLayerProvider layerProvider = new GradientConvolutionLayerProvider(neuralNetworkModel.getLayers().get(0), results, previousResults, 1);

            ConvolutionData error = new ConvolutionData(Collections.singletonList(createMatrix(4, d -> d / 10)), 1);
            LayerTypeCorrectionsVisitor dataVisitor = new LayerTypeCorrectionsVisitor(layerProvider);
            error.accept(dataVisitor);

            DoubleMatrix weightResults = dataVisitor.getCorrection().getCorrectionResults().get(0);
            DoubleMatrix biasResults = dataVisitor.getCorrection().getCorrectionResults().get(1);

            DoubleMatrix errorDatas = ((ConvolutionData) dataVisitor.getNextGradientLayerProvider()).getDatas().get(0);

            DoubleMatrixAssertions.assertMatrices(new DoubleMatrix(3, 3, 69.6, 96.2, 73.2, 111.2, 149.6, 111.2, 73.2, 96.2, 69.6), weightResults);
            DoubleMatrixAssertions.assertMatrices(new DoubleMatrix(1, 1, 13.6), biasResults);
            DoubleMatrixAssertions.assertMatrices(new DoubleMatrix(4, 4, 0.029, 0.062, 0.083, 0.075, 0.099, 0.192, 0.237, 0.198, 0.207, 0.372, 0.417, 0.33, 0.263, 0.446, 0.485, 0.365), errorDatas);
        }

        @Test
        void one_layer_no_padding_with_striding_one_channel() {
            NeuralNetworkModel neuralNetworkModel = ConvolutionNeuralNetworkModelBuilder.init()
                    .useInitializer(InitializerType.ONES)
                    .input(5, 5, 1)
                    .addConvolutionLayer(3, 0, 2, 1, ActivationFunctionType.NONE)
                    .buildConvolutionNetworkModel();

            neuralNetworkModel.getLayers().get(0).getParametersMatrix().get(0).muli(createMatrix(3, d -> d / 100));
            ConvolutionData previousResults = new ConvolutionData(Collections.singletonList(new DoubleMatrix(5, 5, IntStream.range(1, 26).asDoubleStream().toArray())), 1);
            ConvolutionData results = new ConvolutionData(Collections.singletonList(createMatrix(2, d -> d / 2)), 1);
            GradientConvolutionLayerProvider layerProvider = new GradientConvolutionLayerProvider(neuralNetworkModel.getLayers().get(0), results, previousResults, 1);

            ConvolutionData error = new ConvolutionData(Collections.singletonList(createMatrix(2, d -> d / 10)), 1);
            LayerTypeCorrectionsVisitor dataVisitor = new LayerTypeCorrectionsVisitor(layerProvider);
            error.accept(dataVisitor);

            DoubleMatrix weightResults = dataVisitor.getCorrection().getCorrectionResults().get(0);
            DoubleMatrix biasResults = dataVisitor.getCorrection().getCorrectionResults().get(1);

            DoubleMatrix errorDatas = ((ConvolutionData) dataVisitor.getNextGradientLayerProvider()).getDatas().get(0);

            DoubleMatrixAssertions.assertMatrices(new DoubleMatrix(3, 3, 9.2, 10.2, 11.2, 14.2, 15.2, 16.2, 19.2, 20.2, 21.2), weightResults);
            DoubleMatrixAssertions.assertMatrices(new DoubleMatrix(1, 1, 1), biasResults);
            DoubleMatrixAssertions.assertMatrices(new DoubleMatrix(5, 5, 0.001, 0.002, 0.005, 0.004, 0.006, 0.004, 0.005, 0.014, 0.01, 0.012, 0.01, 0.014, 0.036, 0.024, 0.03, 0.012, 0.015, 0.034, 0.02, 0.024, 0.021, 0.024, 0.055, 0.032, 0.036), errorDatas);
        }

        @Test
        void one_layer_no_padding_no_striding_two_channels() {
            NeuralNetworkModel neuralNetworkModel = ConvolutionNeuralNetworkModelBuilder.init()
                    .useInitializer(InitializerType.ONES)
                    .input(4, 4, 2)
                    .addConvolutionLayer(3, 0, 1, 2, ActivationFunctionType.NONE)
                    .buildConvolutionNetworkModel();

            List<DoubleMatrix> parametersMatrix = neuralNetworkModel.getLayers().get(0).getParametersMatrix();
            parametersMatrix.get(0).muli(createMatrix(3, d -> d / 100));
            parametersMatrix.get(1).muli(createMatrix(3, d -> d / 50));
            parametersMatrix.get(2).muli(createMatrix(3, d -> d / 25));
            parametersMatrix.get(3).muli(createMatrix(3, d -> d / 12));

            ConvolutionData previousResults = new ConvolutionData(Arrays.asList(
                    createMatrix(4, DoubleUnaryOperator.identity()),
                    createMatrix(4, d -> d * 2)
            ), 2);
            ConvolutionData results = new ConvolutionData(Arrays.asList(
                    createMatrix(2, d -> d / 2),
                    createMatrix(2, d -> d / 4)
            ), 2);
            GradientConvolutionLayerProvider layerProvider = new GradientConvolutionLayerProvider(neuralNetworkModel.getLayers().get(0), results, previousResults, 1);

            ConvolutionData error = new ConvolutionData(Arrays.asList(
                    createMatrix(2, d -> d / 10),
                    createMatrix(2, d -> d / 5)
            ), 2);

            LayerTypeCorrectionsVisitor dataVisitor = new LayerTypeCorrectionsVisitor(layerProvider);
            error.accept(dataVisitor);

            List<DoubleMatrix> correctionResults = dataVisitor.getCorrection().getCorrectionResults();
            List<DoubleMatrix> weightResults = Arrays.asList(correctionResults.get(0), correctionResults.get(1), correctionResults.get(2), correctionResults.get(3));
            List<DoubleMatrix> biasResults = Arrays.asList(correctionResults.get(4), correctionResults.get(5));

            List<DoubleMatrix> errorDatas = ((ConvolutionData) dataVisitor.getNextGradientLayerProvider()).getDatas();

            DoubleMatrixAssertions.assertMatrices(Arrays.asList(
                    new DoubleMatrix(3, 3, 4.4, 5.4, 6.4, 8.4, 9.4, 10.4, 12.4, 13.4, 14.4),
                    new DoubleMatrix(3, 3, 8.8, 10.8, 12.8, 16.8, 18.8, 20.8, 24.8, 26.8, 28.8),
                    new DoubleMatrix(3, 3, 8.8, 10.8, 12.8, 16.8, 18.8, 20.8, 24.8, 26.8, 28.8),
                    new DoubleMatrix(3, 3, 17.6, 21.6, 25.6, 33.6, 37.6, 41.6, 49.6, 53.6, 57.6)
            ), weightResults);
            DoubleMatrixAssertions.assertMatrices(Arrays.asList(
                    new DoubleMatrix(1, 1, 1),
                    new DoubleMatrix(1, 1, 2)
            ), biasResults);
            DoubleMatrixAssertions.assertMatrices(Arrays.asList(
                    new DoubleMatrix(4, 4, 0.009, 0.036, 0.063, 0.054, 0.063, 0.207, 0.297, 0.216, 0.171, 0.477, 0.567, 0.378, 0.189, 0.468, 0.531, 0.324),
                    new DoubleMatrix(4, 4, 0.018666, 0.074666, 0.130666, 0.112, 0.130666, 0.429333, 0.616, 0.448, 0.354666, 0.989333, 1.176, 0.784, 0.392, 0.970666, 1.101333, 0.672)
            ), errorDatas);
        }
    }

    @Nested
    class MaxPoolingLayer {

        @Test
        void one_layer_no_padding_no_striding_one_channel() {
            NeuralNetworkModel neuralNetworkModel = ConvolutionNeuralNetworkModelBuilder.init()
                    .useInitializer(InitializerType.ONES)
                    .input(5, 5, 1)
                    .addMaxPoolingLayer(3, 0, 1, ActivationFunctionType.NONE)
                    .buildConvolutionNetworkModel();

            MaxPoolingData previousResults = new MaxPoolingData(
                    Collections.singletonList(new DoubleMatrix(5, 5, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 1, 1, 1, 4, 4, 5, 6, 8, 7, 1, 5, 6, 8, 7, 1)),
                    null,
                    null,
                    1
            );
            MaxPoolingData results = new MaxPoolingData(Collections.singletonList(new DoubleMatrix(3, 3, 2, 4, 4, 8, 8, 8, 8, 8, 8)),
                    Collections.singletonList(new DoubleMatrix(3, 3, 0, 3, 3, 2, 2, 2, 2, 2, 2)),
                    Collections.singletonList(new DoubleMatrix(3, 3, 1, 2, 2, 3, 3, 3, 3, 3, 3)),
                    1);
            GradientConvolutionLayerProvider layerProvider = new GradientConvolutionLayerProvider(neuralNetworkModel.getLayers().get(0), results, previousResults, 1);

            MaxPoolingData error = new MaxPoolingData(Collections.singletonList(createMatrix(3, d -> d / 10)),
                    null,
                    null,
                    1);
            LayerTypeCorrectionsVisitor dataVisitor = new LayerTypeCorrectionsVisitor(layerProvider);
            error.accept(dataVisitor);

            DoubleMatrix weightResults = dataVisitor.getCorrection().getCorrectionResults().get(0);

            DoubleMatrix errorDatas = ((MaxPoolingData) dataVisitor.getNextGradientLayerProvider()).getDatas().get(0);

            DoubleMatrixAssertions.assertMatrices(DoubleMatrix.EMPTY, weightResults);
            DoubleMatrixAssertions.assertMatrices(new DoubleMatrix(5, 5, 0, 0, 0, 0, 0, 0.1, 0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 3.9, 0, 0, 0, 0, 0, 0, 0), errorDatas);
        }

        @Test
        void one_layer_with_padding_no_striding_one_channel() {
            NeuralNetworkModel neuralNetworkModel = ConvolutionNeuralNetworkModelBuilder.init()
                    .useInitializer(InitializerType.ONES)
                    .input(4, 4, 1)
                    .addMaxPoolingLayer(3, 1, 1, ActivationFunctionType.NONE)
                    .buildConvolutionNetworkModel();

            MaxPoolingData previousResults = new MaxPoolingData(
                    Collections.singletonList(new DoubleMatrix(4, 4, 5, 4, 2, 3, 7, 4, 2, 3, 9, 1, 1, 5, 1, 2, 3, 8)),
                    null,
                    null,
                    1
            );
            MaxPoolingData results = new MaxPoolingData(Collections.singletonList(new DoubleMatrix(4, 4, 7, 7, 4, 3, 9, 9, 5, 5, 9, 9, 8, 8, 9, 9, 8, 8)),
                    Collections.singletonList(new DoubleMatrix(4, 4, 1, 1, 2, 4, 1, 1, 4, 4, 1, 1, 4, 4, 1, 1, 4, 4)),
                    Collections.singletonList(new DoubleMatrix(4, 4, 2, 2, 1, 1, 3, 3, 4, 4, 3, 3, 4, 4, 3, 3, 4, 4)),
                    1);
            GradientConvolutionLayerProvider layerProvider = new GradientConvolutionLayerProvider(neuralNetworkModel.getLayers().get(0), results, previousResults, 1);

            MaxPoolingData error = new MaxPoolingData(Collections.singletonList(createMatrix(4, d -> d / 10)),
                    null,
                    null,
                    1);
            LayerTypeCorrectionsVisitor dataVisitor = new LayerTypeCorrectionsVisitor(layerProvider);
            error.accept(dataVisitor);

            DoubleMatrix weightResults = dataVisitor.getCorrection().getCorrectionResults().get(0);

            DoubleMatrix errorDatas = ((MaxPoolingData) dataVisitor.getNextGradientLayerProvider()).getDatas().get(0);

            DoubleMatrixAssertions.assertMatrices(DoubleMatrix.EMPTY, weightResults);
            DoubleMatrixAssertions.assertMatrices(new DoubleMatrix(4, 4, 0, 0.3, 0, 0.4, 0.3, 0, 0, 0, 5.7, 0, 0, 0, 0, 0, 0, 6.9), errorDatas);
        }

        @Test
        void one_layer_no_padding_with_striding_one_channel() {
            NeuralNetworkModel neuralNetworkModel = ConvolutionNeuralNetworkModelBuilder.init()
                    .useInitializer(InitializerType.ONES)
                    .input(5, 5, 1)
                    .addMaxPoolingLayer(3, 0, 2, ActivationFunctionType.NONE)
                    .buildConvolutionNetworkModel();

            MaxPoolingData previousResults = new MaxPoolingData(
                    Collections.singletonList(new DoubleMatrix(5, 5, 6, 4, 1, 3, 5, 5, 5, 4, 2, 3, 2, 7, 4, 2, 3, 3, 9, 1, 1, 5, 7, 1, 2, 3, 8)),
                    null,
                    null,
                    1
            );
            MaxPoolingData results = new MaxPoolingData(Collections.singletonList(new DoubleMatrix(2, 2, 7, 5, 9, 8)),
                    Collections.singletonList(new DoubleMatrix(2, 2, 1, 0, 1, 4)),
                    Collections.singletonList(new DoubleMatrix(2, 2, 2, 1, 3, 4)),
                    1);
            GradientConvolutionLayerProvider layerProvider = new GradientConvolutionLayerProvider(neuralNetworkModel.getLayers().get(0), results, previousResults, 1);

            MaxPoolingData error = new MaxPoolingData(Collections.singletonList(new DoubleMatrix(2, 2, 0.1, 0.2, 0.5, 0.6)),
                    null,
                    null,
                    1);
            LayerTypeCorrectionsVisitor dataVisitor = new LayerTypeCorrectionsVisitor(layerProvider);
            error.accept(dataVisitor);

            DoubleMatrix weightResults = dataVisitor.getCorrection().getCorrectionResults().get(0);

            DoubleMatrix errorDatas = ((MaxPoolingData) dataVisitor.getNextGradientLayerProvider()).getDatas().get(0);

            DoubleMatrixAssertions.assertMatrices(DoubleMatrix.EMPTY, weightResults);
            DoubleMatrixAssertions.assertMatrices(new DoubleMatrix(5, 5, 0, 0, 0, 0, 0, 0.2, 0, 0, 0, 0, 0, 0.1, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0, 0.6), errorDatas);
        }
    }

    @Nested
    class AveragePoolingLayer {

        @Test
        void one_layer_no_padding_no_striding_one_channel() {
            NeuralNetworkModel neuralNetworkModel = ConvolutionNeuralNetworkModelBuilder.init()
                    .useInitializer(InitializerType.ONES)
                    .input(5, 5, 1)
                    .addAveragePoolingLayer(3, 0, 1, ActivationFunctionType.NONE)
                    .buildConvolutionNetworkModel();

            AveragePoolingData previousResults = new AveragePoolingData(
                    Collections.singletonList(DoubleMatrix.zeros(5, 5))
                    , 1
            );
            AveragePoolingData results = new AveragePoolingData(Collections.singletonList(DoubleMatrix.zeros(3, 3)), 1);
            GradientConvolutionLayerProvider layerProvider = new GradientConvolutionLayerProvider(neuralNetworkModel.getLayers().get(0), results, previousResults, 1);

            AveragePoolingData error = new AveragePoolingData(Collections.singletonList(createMatrix(3, d -> d / 10)), 1);
            LayerTypeCorrectionsVisitor dataVisitor = new LayerTypeCorrectionsVisitor(layerProvider);
            error.accept(dataVisitor);

            DoubleMatrix weightResults = dataVisitor.getCorrection().getCorrectionResults().get(0);

            DoubleMatrix errorDatas = ((AveragePoolingData) dataVisitor.getNextGradientLayerProvider()).getDatas().get(0);

            DoubleMatrixAssertions.assertMatrices(DoubleMatrix.EMPTY, weightResults);
            DoubleMatrixAssertions.assertMatrices(new DoubleMatrix(5, 5, 0.011111, 0.033333, 0.066666, 0.055555, 0.033333, 0.055555, 0.133333, 0.233333, 0.177777, 0.1, 0.133333, 0.3, 0.5, 0.366666, 0.2, 0.122222, 0.266666, 0.433333, 0.311111, 0.166666, 0.077777, 0.166666, 0.266666, 0.188888, 0.1), errorDatas);
        }

        @Test
        void one_layer_with_padding_no_striding_one_channel() {
            NeuralNetworkModel neuralNetworkModel = ConvolutionNeuralNetworkModelBuilder.init()
                    .useInitializer(InitializerType.ONES)
                    .input(4, 4, 1)
                    .addAveragePoolingLayer(3, 1, 1, ActivationFunctionType.NONE)
                    .buildConvolutionNetworkModel();

            AveragePoolingData previousResults = new AveragePoolingData(
                    Collections.singletonList(DoubleMatrix.zeros(4, 4)), 1
            );
            AveragePoolingData results = new AveragePoolingData(Collections.singletonList(DoubleMatrix.zeros(3, 3)), 1);
            GradientConvolutionLayerProvider layerProvider = new GradientConvolutionLayerProvider(neuralNetworkModel.getLayers().get(0), results, previousResults, 1);

            AveragePoolingData error = new AveragePoolingData(Collections.singletonList(createMatrix(4, d -> d / 10)), 1);
            LayerTypeCorrectionsVisitor dataVisitor = new LayerTypeCorrectionsVisitor(layerProvider);
            error.accept(dataVisitor);

            DoubleMatrix weightResults = dataVisitor.getCorrection().getCorrectionResults().get(0);

            DoubleMatrix errorDatas = ((AveragePoolingData) dataVisitor.getNextGradientLayerProvider()).getDatas().get(0);

            DoubleMatrixAssertions.assertMatrices(DoubleMatrix.EMPTY, weightResults);
            DoubleMatrixAssertions.assertMatrices(new DoubleMatrix(4, 4, 0.15555, 0.26666, 0.33333, 0.24444, 0.36666, 0.6, 0.7, 0.5, 0.63333, 1, 1.1, 0.76666, 0.51111, 0.8, 0.86666, 0.6), errorDatas);
        }

        @Test
        void one_layer_no_padding_with_striding_one_channel() {
            NeuralNetworkModel neuralNetworkModel = ConvolutionNeuralNetworkModelBuilder.init()
                    .useInitializer(InitializerType.ONES)
                    .input(5, 5, 1)
                    .addAveragePoolingLayer(3, 0, 2, ActivationFunctionType.NONE)
                    .buildConvolutionNetworkModel();

            AveragePoolingData previousResults = new AveragePoolingData(
                    Collections.singletonList(DoubleMatrix.zeros(5, 5)),
                    1
            );
            AveragePoolingData results = new AveragePoolingData(Collections.singletonList(DoubleMatrix.zeros(3, 3)), 1);
            GradientConvolutionLayerProvider layerProvider = new GradientConvolutionLayerProvider(neuralNetworkModel.getLayers().get(0), results, previousResults, 1);

            AveragePoolingData error = new AveragePoolingData(Collections.singletonList(createMatrix(2, d -> d / 10)), 1);
            LayerTypeCorrectionsVisitor dataVisitor = new LayerTypeCorrectionsVisitor(layerProvider);
            error.accept(dataVisitor);

            DoubleMatrix weightResults = dataVisitor.getCorrection().getCorrectionResults().get(0);

            DoubleMatrix errorDatas = ((AveragePoolingData) dataVisitor.getNextGradientLayerProvider()).getDatas().get(0);

            DoubleMatrixAssertions.assertMatrices(DoubleMatrix.EMPTY, weightResults);
            DoubleMatrixAssertions.assertMatrices(new DoubleMatrix(5, 5, 0.01111, 0.01111, 0.03333, 0.02222, 0.02222, 0.01111, 0.01111, 0.03333, 0.02222, 0.02222, 0.04444, 0.04444, 0.11111, 0.06666, 0.06666, 0.03333, 0.03333, 0.07777, 0.04444, 0.04444, 0.03333, 0.03333, 0.07777, 0.04444, 0.04444), errorDatas);
        }

    }

}