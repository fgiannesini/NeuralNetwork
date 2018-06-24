package com.fgiannesini.neuralnetwork.computer;

import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import org.jblas.FloatMatrix;

import java.util.List;
import java.util.stream.Collectors;

public class OutputComputer {

    private final List<LayerComputer> layerComputers;

    OutputComputer(NeuralNetworkModel model) {
        layerComputers = model.getLayers().stream()
                .map(LayerComputer::new)
                .collect(Collectors.toList());
    }

    public float[] compute(float[] input) {
        return compute(new FloatMatrix(input)).toArray();
    }

    public float[][] compute(float[][] input) {
        FloatMatrix inputMatrix = new FloatMatrix(input).transpose();
        return compute(inputMatrix).transpose().toArray2();
    }

    private FloatMatrix compute(FloatMatrix currentMatrix) {
        for (LayerComputer layerComputer : layerComputers) {
            currentMatrix = layerComputer.compute(currentMatrix);
        }
        return currentMatrix;
    }
}
