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
    },
    //Used for RELU
    UNIFORM {
        @Override
        public Initializer getInitializer() {
            return new UniformInitializer();
        }
    },
    //Used for TANH
    XAVIER_IN {
        @Override
        public Initializer getInitializer() {
            return new XavierInInitializer();
        }
    },
    XAVIER {
        @Override
        public Initializer getInitializer() {
            return new XavierInitializer();
        }
    };

    public abstract Initializer getInitializer();
}
