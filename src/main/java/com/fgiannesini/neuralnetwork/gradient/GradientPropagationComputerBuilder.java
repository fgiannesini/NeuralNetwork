package com.fgiannesini.neuralnetwork.gradient;

import com.fgiannesini.neuralnetwork.computer.OutputComputer;
import com.fgiannesini.neuralnetwork.computer.OutputComputerBuilder;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;

public class GradientPropagationComputerBuilder {

    private NeuralNetworkModel neuralNetworkModel;

    private GradientPropagationComputerBuilder() {
    }

    public static GradientPropagationComputerBuilder init() {
        return new GradientPropagationComputerBuilder();
    }

    public GradientPropagationComputerBuilder withModel(NeuralNetworkModel neuralNetworkModel) {
        this.neuralNetworkModel = neuralNetworkModel;
        return this;
    }

    public GradientPropagationLearner build() {
        if (neuralNetworkModel == null) {
            throw new IllegalArgumentException("NeuralNetworkModel missing");
        }
        OutputComputer outputComputer = OutputComputerBuilder.init()
                .withModel(neuralNetworkModel)
                .build();
        return new GradientPropagationLearner(neuralNetworkModel);
    }

}
