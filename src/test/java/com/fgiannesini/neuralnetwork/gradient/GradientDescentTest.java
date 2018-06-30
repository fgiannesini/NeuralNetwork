package com.fgiannesini.neuralnetwork.gradient;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import com.fgiannesini.neuralnetwork.initializer.InitializerType;
import com.fgiannesini.neuralnetwork.model.Layer;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModelBuilder;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.function.Executable;

import java.util.ArrayList;
import java.util.List;

class GradientDescentTest {

  @Test
  void learn_on_vector_with_one_hidden_layer_learning_is_optimal() {
    NeuralNetworkModel neuralNetworkModel = NeuralNetworkModelBuilder.init()
      .useInitializer(InitializerType.ONES)
            .input(3)
      .addLayer(4, ActivationFunctionType.NONE)
            .output(2, ActivationFunctionType.NONE)
      .build();
    GradientDescent gradientDescent = new GradientDescent(neuralNetworkModel, 0.01f);

    double[] input = new double[]{1f, 2f, 3f};
    double[] output = new double[]{29f, 29f};

    NeuralNetworkModel optimizedNeuralNetworkModel = gradientDescent.learn(input, output);

    checkSameNeuralNetworks(neuralNetworkModel, optimizedNeuralNetworkModel);
  }

  @Test
  void learn_on_matrix_with_one_hidden_layer_learning_is_optimal() {
    NeuralNetworkModel neuralNetworkModel = NeuralNetworkModelBuilder.init()
      .useInitializer(InitializerType.ONES)
            .input(3)
      .addLayer(4, ActivationFunctionType.NONE)
            .output(2, ActivationFunctionType.NONE)
      .build();
    GradientDescent gradientDescent = new GradientDescent(neuralNetworkModel, 0.01f);

    double[][] input = new double[][]{
      {1f, 2f, 3f},
      {3f, 2f, 1f}
    };
    double[][] output = new double[][]{
      {29f, 29f},
      {29f, 29f}
    };

    NeuralNetworkModel optimizedNeuralNetworkModel = gradientDescent.learn(input, output);

    checkSameNeuralNetworks(neuralNetworkModel, optimizedNeuralNetworkModel);

  }

  @Test
  void learn_on_vector_with_four_hidden_layers_learning_is_optimal() {
    NeuralNetworkModel neuralNetworkModel = NeuralNetworkModelBuilder.init()
      .useInitializer(InitializerType.ONES)
            .input(3)
      .addLayer(4, ActivationFunctionType.NONE)
      .addLayer(5, ActivationFunctionType.NONE)
      .addLayer(6, ActivationFunctionType.NONE)
      .addLayer(5, ActivationFunctionType.NONE)
            .output(2, ActivationFunctionType.NONE)
      .build();
    GradientDescent gradientDescent = new GradientDescent(neuralNetworkModel, 0.01f);

    double[] input = new double[]{1f, 2f, 3f};
    double[] output = new double[]{4386f, 4386f};

    NeuralNetworkModel optimizedNeuralNetworkModel = gradientDescent.learn(input, output);

    checkSameNeuralNetworks(neuralNetworkModel, optimizedNeuralNetworkModel);
  }

  @Test
  void learn_on_matrix_with_four_hidden_layers_learning_is_optimal() {
    NeuralNetworkModel neuralNetworkModel = NeuralNetworkModelBuilder.init()
      .useInitializer(InitializerType.ONES)
            .input(3)
      .addLayer(4, ActivationFunctionType.NONE)
      .addLayer(5, ActivationFunctionType.NONE)
      .addLayer(6, ActivationFunctionType.NONE)
      .addLayer(5, ActivationFunctionType.NONE)
            .output(2, ActivationFunctionType.NONE)
      .build();
    GradientDescent gradientDescent = new GradientDescent(neuralNetworkModel, 0.01f);

    double[][] input = new double[][]{
      {1f, 2f, 3f},
      {3f, 2f, 1f}
    };
    double[][] output = new double[][]{
      {4386f, 4386f},
      {4386f, 4386f}
    };

    NeuralNetworkModel optimizedNeuralNetworkModel = gradientDescent.learn(input, output);

    checkSameNeuralNetworks(neuralNetworkModel, optimizedNeuralNetworkModel);
  }

  @Test
  void learn_on_matrix_with_four_hidden_layers_with_activation_functions_learning_is_optimal() {
    NeuralNetworkModel neuralNetworkModel = NeuralNetworkModelBuilder.init()
      .useInitializer(InitializerType.ONES)
            .input(3)
      .addLayer(4)
      .addLayer(5)
      .addLayer(6)
      .addLayer(5)
            .output(2)
      .build();
    GradientDescent gradientDescent = new GradientDescent(neuralNetworkModel, 0.01f);

    double[][] input = new double[][]{
      {1f, 2f, 3f},
      {3f, 2f, 1f}
    };
    double[][] output = new double[][]{
      {1f, 1f},
      {1f, 1f}
    };

    NeuralNetworkModel optimizedNeuralNetworkModel = gradientDescent.learn(input, output);

    checkSameNeuralNetworks(neuralNetworkModel, optimizedNeuralNetworkModel);
  }

  private void checkSameNeuralNetworks(NeuralNetworkModel neuralNetworkModel, NeuralNetworkModel optimizedNeuralNetworkModel) {
    Assertions.assertAll(
      () -> Assertions.assertEquals(neuralNetworkModel.getOutputSize(), optimizedNeuralNetworkModel.getOutputSize()),
      () -> Assertions.assertEquals(neuralNetworkModel.getInputSize(), optimizedNeuralNetworkModel.getInputSize())
    );

    List<Layer> layers = neuralNetworkModel.getLayers();
    List<Layer> optimizedLayers = optimizedNeuralNetworkModel.getLayers();
    List<Executable> executables = new ArrayList<>();
    for (int i = 1; i < optimizedLayers.size(); i++) {
      Layer layer = layers.get(i);
      Layer optimizedLayer = optimizedLayers.get(i);
      executables.add(() -> Assertions.assertEquals(layer.getWeightMatrix(), optimizedLayer.getWeightMatrix()));
      executables.add(() -> Assertions.assertEquals(layer.getBiasMatrix(), optimizedLayer.getBiasMatrix()));
    }
    Assertions.assertAll(executables);
  }
}