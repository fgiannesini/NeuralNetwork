package com.fgiannesini.neuralnetwork.normalizer;

public enum NormalizerType {

    NONE {
        @Override
        public INormalizer get() {
            return new NoneNormalizer();
        }
    },
    MEAN_AND_DEVIATION {
        @Override
        public INormalizer get() {
            return new MeanAndDeviationNormalizer();
        }
    };

    public abstract INormalizer get();
}
