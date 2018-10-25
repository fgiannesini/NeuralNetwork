package com.fgiannesini.neuralnetwork.serializer;

import java.io.*;

public class Serializer {

    static final String SERIALIZATION_EXTENSION = ".serialized";

    private Serializer() {
    }

    public static Serializer get() {
        return new Serializer();
    }

    public <T extends Serializable> void serialize(T t, String name) {
        File file = new File(name + SERIALIZATION_EXTENSION);
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(file))) {
            oos.writeObject(t);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public <T extends Serializable> T deserialize(String name) {
        File file = new File(name + SERIALIZATION_EXTENSION);
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file))) {
            return (T) ois.readObject();
        } catch (IOException | ClassNotFoundException e) {
            throw new RuntimeException(e);
        }
    }
}
