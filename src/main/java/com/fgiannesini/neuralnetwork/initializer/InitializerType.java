package com.fgiannesini.neuralnetwork.initializer;

public enum InitializerType {

    ZEROS {
        @Override
        public Initializer getInitializer() {
            return new ZerosInitializer();
        }
    },
    RANDOM {
        @Override
        public Initializer getInitializer() {
            return new RandomInitializer();
        }
    },
    ONES {
        @Override
        public Initializer getInitializer() {
            return new OnesInitializer();
        }
    };

    public abstract Initializer getInitializer();
}
