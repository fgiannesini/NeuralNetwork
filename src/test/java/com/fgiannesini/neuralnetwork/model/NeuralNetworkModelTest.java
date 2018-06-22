package com.fgiannesini.neuralnetwork.model;

import org.junit.jupiter.api.Test;

class NeuralNetworkModelTest {

  @Test
  void create_Network_Model() {
    NeuralNetworkModel neuralNetworkConfig = NeuralNetworkModelBuilder.init()
      .inputSize(10)
      .addLayer(5)
      .addLayer(7)
      .outputSize(10)
      .build();


  }

}