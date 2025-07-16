package dev.neuronic.net;

import java.util.Arrays;

/**
 * Represents the shape of a tensor (multi-dimensional array).
 * This class is immutable.
 */
public record Shape(int[] dims) {

    public static Shape vector(int size) {
        return new Shape(new int[]{size});
    }

    public static Shape sequence(int sequenceLength, int features) {
        return new Shape(new int[]{sequenceLength, features});
    }
    
    public static Shape of(int... dimensions) {
        return new Shape(dimensions);
    }

    public int rank() {
        return dims.length;
    }

    public int dim(int i) {
        return dims[i];
    }

    public int toFlatSize() {
        int size = 1;
        for (int dim : dims) {
            size *= dim;
        }
        return size;
    }

    @Override
    public String toString() {
        return "Shape" + Arrays.toString(dims);
    }
}
