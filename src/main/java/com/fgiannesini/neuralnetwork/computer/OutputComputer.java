package com.fgiannesini.neuralnetwork.computer;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionApplyer;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import org.jblas.FloatMatrix;

import java.util.List;
import java.util.stream.Collectors;

public class OutputComputer {

    private final List<LayerComputer> layerComputers;

    OutputComputer(NeuralNetworkModel model, ActivationFunctionApplyer activationFunctionApplyer) {
        layerComputers = model.getLayers().stream()
                .map(layer -> new LayerComputer(layer, activationFunctionApplyer))
                .collect(Collectors.toList());
    }

    public float[] compute(float[] input) {
        FloatMatrix currentMatrix = new FloatMatrix(input);
        for (LayerComputer layerComputer : layerComputers) {
            currentMatrix = layerComputer.compute(currentMatrix);
        }
        return currentMatrix.data;
    }
}
