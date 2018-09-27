package com.fgiannesini.neuralnetwork.normalizer;

import com.fgiannesini.neuralnetwork.computer.LayerTypeData;

public interface INormalizer {

    LayerTypeData normalize(LayerTypeData input);
}
