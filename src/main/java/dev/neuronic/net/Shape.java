package dev.neuronic.net;

import java.util.Arrays;

/**
 * Represents the shape of a tensor in the neural network.
 * 
 * <p>This class provides a type-safe way to represent and manipulate tensor dimensions,
 * enabling proper shape inference and validation throughout the network.
 * 
 * <p><b>Common usage patterns:</b>
 * <ul>
 *   <li>1D Vector: Shape.of(784) for flattened MNIST</li>
 *   <li>2D Sequence: Shape.of(50, 128) for 50 timesteps × 128 features</li>
 *   <li>3D Image: Shape.of(28, 28, 1) for MNIST image</li>
 *   <li>4D Batch: Shape.of(32, 28, 28, 1) for batch of 32 MNIST images</li>
 * </ul>
 */
public final class Shape {
    private final int[] dimensions;
    
    private Shape(int... dimensions) {
        if (dimensions == null || dimensions.length == 0) {
            throw new IllegalArgumentException("Shape must have at least one dimension");
        }
        for (int i = 0; i < dimensions.length; i++) {
            if (dimensions[i] <= 0) {
                throw new IllegalArgumentException("Dimension " + i + " must be positive, got: " + dimensions[i]);
            }
        }
        this.dimensions = dimensions.clone();
    }
    
    // Factory methods
    public static Shape of(int... dimensions) {
        return new Shape(dimensions);
    }
    
    public static Shape vector(int size) {
        return new Shape(size);
    }
    
    public static Shape matrix(int rows, int cols) {
        return new Shape(rows, cols);
    }
    
    public static Shape sequence(int seqLen, int features) {
        return new Shape(seqLen, features);
    }
    
    public static Shape image(int height, int width, int channels) {
        return new Shape(height, width, channels);
    }
    
    // For backward compatibility with existing code
    public static Shape fromFlatSize(int size) {
        return new Shape(size);
    }
    
    // Accessors
    public int rank() { 
        return dimensions.length; 
    }
    
    public int dim(int axis) {
        if (axis < 0) {
            axis = dimensions.length + axis; // Support negative indexing
        }
        if (axis < 0 || axis >= dimensions.length) {
            throw new IndexOutOfBoundsException("Axis " + axis + " out of bounds for shape with rank " + dimensions.length);
        }
        return dimensions[axis];
    }
    
    public int[] dims() {
        return dimensions.clone();
    }
    
    public int size() {
        int size = 1;
        for (int dim : dimensions) {
            size *= dim;
        }
        return size;
    }
    
    // For backward compatibility
    public int toFlatSize() {
        return size();
    }
    
    // Shape manipulation
    public Shape flatten() {
        return new Shape(size());
    }
    
    public Shape reshape(int... newDimensions) {
        int newSize = 1;
        int inferredAxis = -1;
        
        for (int i = 0; i < newDimensions.length; i++) {
            if (newDimensions[i] == -1) {
                if (inferredAxis != -1) {
                    throw new IllegalArgumentException("Can only infer one dimension");
                }
                inferredAxis = i;
            } else if (newDimensions[i] > 0) {
                newSize *= newDimensions[i];
            } else {
                throw new IllegalArgumentException("Invalid dimension: " + newDimensions[i]);
            }
        }
        
        if (inferredAxis != -1) {
            newDimensions[inferredAxis] = size() / newSize;
            if (newDimensions[inferredAxis] * newSize != size()) {
                throw new IllegalArgumentException("Cannot reshape " + this + " to " + Arrays.toString(newDimensions));
            }
        } else if (newSize != size()) {
            throw new IllegalArgumentException("Cannot reshape " + this + " to " + Arrays.toString(newDimensions) + 
                                             " (size mismatch: " + size() + " vs " + newSize + ")");
        }
        
        return new Shape(newDimensions);
    }
    
    // Utility methods
    public boolean isVector() {
        return dimensions.length == 1;
    }
    
    public boolean isMatrix() {
        return dimensions.length == 2;
    }
    
    public boolean isCompatibleWith(Shape other) {
        return size() == other.size();
    }
    
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (!(obj instanceof Shape)) return false;
        Shape other = (Shape) obj;
        return Arrays.equals(dimensions, other.dimensions);
    }
    
    @Override
    public int hashCode() {
        return Arrays.hashCode(dimensions);
    }
    
    @Override
    public String toString() {
        return Arrays.toString(dimensions);
    }
}