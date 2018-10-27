package com.fgiannesini.neuralnetwork.serializer;

import java.io.*;

public class Serializer {

    private Serializer() {
    }

    public static Serializer get() {
        return new Serializer();
    }

    public <T extends Serializable> void serialize(T t, File file) {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(file))) {
            oos.writeObject(t);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public <T extends Serializable> T deserialize(File file) {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file))) {
            return (T) ois.readObject();
        } catch (IOException | ClassNotFoundException e) {
            throw new RuntimeException(e);
        }
    }
}
