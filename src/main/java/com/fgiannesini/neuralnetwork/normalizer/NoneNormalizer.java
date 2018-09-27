package com.fgiannesini.neuralnetwork.normalizer;

import com.fgiannesini.neuralnetwork.computer.LayerTypeData;

public class NoneNormalizer implements INormalizer {

    @Override
    public LayerTypeData normalize(LayerTypeData input) {
        return input;
    }
}
