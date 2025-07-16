package dev.neuronic.net.simple;

import java.util.Arrays;
import java.util.List;

/**
 * A batch of training data containing inputs and targets.
 * 
 * <p>This class represents a mini-batch used during training, containing
 * aligned arrays of inputs and their corresponding targets.
 * 
 * <p><b>Creating batches:</b>
 * <pre>{@code
 * // Single example
 * DataBatch<float[], Integer> single = DataBatch.single(
 *     new float[]{1.0f, 2.0f, 3.0f}, 
 *     7
 * );
 * 
 * // Multiple examples
 * DataBatch<String[], String> batch = DataBatch.of(
 *     new String[][]{{"the", "cat"}, {"a", "dog"}},
 *     new String[]{"sat", "ran"}
 * );
 * 
 * // From lists
 * DataBatch<float[], Integer> fromLists = DataBatch.fromLists(
 *     inputList,
 *     targetList
 * );
 * }</pre>
 * 
 * @param <I> input type
 * @param <T> target type
 */
public class DataBatch<I, T> {
    private final I[] inputs;
    private final T[] targets;
    
    /**
     * Create a new data batch.
     * 
     * @param inputs array of inputs
     * @param targets array of targets (must be same length as inputs)
     * @throws IllegalArgumentException if lengths don't match
     */
    public DataBatch(I[] inputs, T[] targets) {
        if (inputs.length != targets.length) {
            throw new IllegalArgumentException(
                "Inputs and targets must have same length: " + 
                inputs.length + " vs " + targets.length);
        }
        this.inputs = inputs;
        this.targets = targets;
    }
    
    /**
     * Get the input data.
     * 
     * @return array of inputs
     */
    public I[] getInputs() { 
        return inputs; 
    }
    
    /**
     * Get the target data.
     * 
     * @return array of targets
     */
    public T[] getTargets() { 
        return targets; 
    }
    
    /**
     * Get the batch size.
     * 
     * @return number of examples in this batch
     */
    public int size() { 
        return inputs.length; 
    }
    
    /**
     * Check if batch is empty.
     * 
     * @return true if batch has no examples
     */
    public boolean isEmpty() { 
        return inputs.length == 0; 
    }
    
    /**
     * Get a specific example from the batch.
     * 
     * @param index example index
     * @return pair of (input, target) at given index
     */
    public Pair<I, T> get(int index) {
        if (index < 0 || index >= inputs.length) {
            throw new IndexOutOfBoundsException(
                "Index " + index + " out of bounds for batch size " + inputs.length);
        }
        return new Pair<>(inputs[index], targets[index]);
    }
    
    // ===============================
    // FACTORY METHODS
    // ===============================
    
    /**
     * Create a batch with a single example.
     * 
     * @param input single input
     * @param target single target
     * @return batch containing one example
     */
    @SuppressWarnings("unchecked")
    public static <I, T> DataBatch<I, T> single(I input, T target) {
        I[] inputs = (I[]) new Object[]{input};
        T[] targets = (T[]) new Object[]{target};
        return new DataBatch<>(inputs, targets);
    }
    
    /**
     * Create a batch from arrays.
     * 
     * @param inputs array of inputs
     * @param targets array of targets
     * @return new batch
     */
    public static <I, T> DataBatch<I, T> of(I[] inputs, T[] targets) {
        return new DataBatch<>(inputs, targets);
    }
    
    /**
     * Create a batch from lists.
     * 
     * @param inputs list of inputs
     * @param targets list of targets
     * @return new batch
     */
    @SuppressWarnings("unchecked")
    public static <I, T> DataBatch<I, T> fromLists(List<I> inputs, List<T> targets) {
        if (inputs.size() != targets.size()) {
            throw new IllegalArgumentException(
                "Lists must have same size: " + 
                inputs.size() + " vs " + targets.size());
        }
        
        I[] inputArray = (I[]) inputs.toArray();
        T[] targetArray = (T[]) targets.toArray();
        
        return new DataBatch<>(inputArray, targetArray);
    }
    
    /**
     * Create an empty batch.
     * 
     * @return empty batch
     */
    @SuppressWarnings("unchecked")
    public static <I, T> DataBatch<I, T> empty() {
        return new DataBatch<>((I[]) new Object[0], (T[]) new Object[0]);
    }
    
    /**
     * Simple pair class for returning individual examples.
     */
    public static class Pair<I, T> {
        public final I input;
        public final T target;
        
        public Pair(I input, T target) {
            this.input = input;
            this.target = target;
        }
    }
    
    @Override
    public String toString() {
        return "DataBatch[size=" + size() + "]";
    }
}