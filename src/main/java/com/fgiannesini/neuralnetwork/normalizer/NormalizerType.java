package com.fgiannesini.neuralnetwork.normalizer;

import com.fgiannesini.neuralnetwork.normalizer.meandeviation.MeanAndDeviationNormalizer;
import com.fgiannesini.neuralnetwork.normalizer.meandeviation.MeanDeviationProvider;

import java.util.Arrays;

public enum NormalizerType {

    NONE {
        public INormalizer get(Object... args) {
            return new NoneNormalizer();
        }
    },
    MEAN_AND_DEVIATION {
        public INormalizer get(Object... args) {
            MeanDeviationProvider meanDeviationProvider = Arrays.stream(args)
                    .filter(arg -> arg instanceof MeanDeviationProvider)
                    .map(MeanDeviationProvider.class::cast)
                    .findAny()
                    .orElseThrow(() -> new IllegalArgumentException(MEAN_AND_DEVIATION + " should have a arument" + MeanDeviationProvider.class.getSimpleName()));
            return new MeanAndDeviationNormalizer(meanDeviationProvider);
        }
    };

    public abstract INormalizer get(Object... args);
}
