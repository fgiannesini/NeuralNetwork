package com.fgiannesini.neuralnetwork.computer;

import com.fgiannesini.neuralnetwork.model.LayerType;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

class LayerComputerBuilderTest {

    @Test
    void instanciate_batch_norm() {
        LayerComputer layerComputer = LayerComputerBuilder.init()
                .withLayerType(LayerType.BATCH_NORM)
                .build();
        Assertions.assertTrue(layerComputer instanceof BatchNormLayerComputer);
    }

    @Test
    void instanciate_weight_bias() {
        LayerComputer layerComputer = LayerComputerBuilder.init()
                .withLayerType(LayerType.WEIGHT_BIAS)
                .build();
        Assertions.assertTrue(layerComputer instanceof WeighBiasLayerComputer);
    }

}