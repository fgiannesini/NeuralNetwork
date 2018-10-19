package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import com.fgiannesini.neuralnetwork.assertions.DoubleMatrixAssertions;
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
            ConvolutionData previousResults = new ConvolutionData(Collections.singletonList(createMatrix(5, DoubleUnaryOperator.identity())));
            ConvolutionData results = new ConvolutionData(Collections.singletonList(createMatrix(3, DoubleUnaryOperator.identity())));
            GradientConvolutionLayerProvider layerProvider = new GradientConvolutionLayerProvider(neuralNetworkModel.getLayers().get(0), results, previousResults, 1);

            ConvolutionData error = new ConvolutionData(Collections.singletonList(createMatrix(3, d -> d / 10)));
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
            ConvolutionData previousResults = new ConvolutionData(Collections.singletonList(createMatrix(4, DoubleUnaryOperator.identity())));
            ConvolutionData results = new ConvolutionData(Collections.singletonList(createMatrix(4, d -> d / 2)));
            GradientConvolutionLayerProvider layerProvider = new GradientConvolutionLayerProvider(neuralNetworkModel.getLayers().get(0), results, previousResults, 1);

            ConvolutionData error = new ConvolutionData(Collections.singletonList(createMatrix(4, d -> d / 10)));
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
            ConvolutionData previousResults = new ConvolutionData(Collections.singletonList(new DoubleMatrix(5, 5, IntStream.range(1, 26).asDoubleStream().toArray())));
            ConvolutionData results = new ConvolutionData(Collections.singletonList(createMatrix(2, d -> d / 2)));
            GradientConvolutionLayerProvider layerProvider = new GradientConvolutionLayerProvider(neuralNetworkModel.getLayers().get(0), results, previousResults, 1);

            ConvolutionData error = new ConvolutionData(Collections.singletonList(createMatrix(2, d -> d / 10)));
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
            ));
            ConvolutionData results = new ConvolutionData(Arrays.asList(
                    createMatrix(2, d -> d / 2),
                    createMatrix(2, d -> d / 4)
            ));
            GradientConvolutionLayerProvider layerProvider = new GradientConvolutionLayerProvider(neuralNetworkModel.getLayers().get(0), results, previousResults, 1);

            ConvolutionData error = new ConvolutionData(Arrays.asList(
                    createMatrix(2, d -> d / 10),
                    createMatrix(2, d -> d / 5)
            ));

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
                    null
            );
            MaxPoolingData results = new MaxPoolingData(Collections.singletonList(new DoubleMatrix(3, 3, 2, 4, 4, 8, 8, 8, 8, 8, 8)),
                    Collections.singletonList(new DoubleMatrix(3, 3, 0, 3, 3, 2, 2, 2, 2, 2, 2)),
                    Collections.singletonList(new DoubleMatrix(3, 3, 1, 2, 2, 3, 3, 3, 3, 3, 3)));
            GradientConvolutionLayerProvider layerProvider = new GradientConvolutionLayerProvider(neuralNetworkModel.getLayers().get(0), results, previousResults, 1);

            MaxPoolingData error = new MaxPoolingData(Collections.singletonList(createMatrix(3, d -> d / 10)),
                    null,
                    null);
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
                    .addConvolutionLayer(3, 1, 1, 1, ActivationFunctionType.NONE)
                    .buildConvolutionNetworkModel();

            neuralNetworkModel.getLayers().get(0).getParametersMatrix().get(0).muli(createMatrix(3, d -> d / 100));
            ConvolutionData previousResults = new ConvolutionData(Collections.singletonList(createMatrix(4, DoubleUnaryOperator.identity())));
            ConvolutionData results = new ConvolutionData(Collections.singletonList(createMatrix(4, d -> d / 2)));
            GradientConvolutionLayerProvider layerProvider = new GradientConvolutionLayerProvider(neuralNetworkModel.getLayers().get(0), results, previousResults, 1);

            ConvolutionData error = new ConvolutionData(Collections.singletonList(createMatrix(4, d -> d / 10)));
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
            ConvolutionData previousResults = new ConvolutionData(Collections.singletonList(new DoubleMatrix(5, 5, IntStream.range(1, 26).asDoubleStream().toArray())));
            ConvolutionData results = new ConvolutionData(Collections.singletonList(createMatrix(2, d -> d / 2)));
            GradientConvolutionLayerProvider layerProvider = new GradientConvolutionLayerProvider(neuralNetworkModel.getLayers().get(0), results, previousResults, 1);

            ConvolutionData error = new ConvolutionData(Collections.singletonList(createMatrix(2, d -> d / 10)));
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
            ));
            ConvolutionData results = new ConvolutionData(Arrays.asList(
                    createMatrix(2, d -> d / 2),
                    createMatrix(2, d -> d / 4)
            ));
            GradientConvolutionLayerProvider layerProvider = new GradientConvolutionLayerProvider(neuralNetworkModel.getLayers().get(0), results, previousResults, 1);

            ConvolutionData error = new ConvolutionData(Arrays.asList(
                    createMatrix(2, d -> d / 10),
                    createMatrix(2, d -> d / 5)
            ));

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
    class AveragePoolingLayer {


    }

}