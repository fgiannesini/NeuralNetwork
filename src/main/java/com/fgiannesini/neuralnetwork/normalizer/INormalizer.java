package com.fgiannesini.neuralnetwork.normalizer;

import com.fgiannesini.neuralnetwork.computer.data.LayerTypeData;

public interface INormalizer {

    LayerTypeData normalize(LayerTypeData input);
}
