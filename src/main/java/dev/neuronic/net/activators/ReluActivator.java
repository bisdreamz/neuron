package dev.neuronic.net.activators;

import dev.neuronic.net.math.Parallelization;
import java.util.concurrent.ExecutorService;

/**
 * Rectified Linear Unit (ReLU) activation function.
 *
 * <p>ReLU(x) = max(0, x)
 *
 * <p>The most widely used activation function in modern neural networks.
 * Simple, efficient, and helps avoid vanishing gradient problems.
 *
 * <p>Derivative: 1 if x > 0, else 0
 */
public final class ReluActivator implements Activator {

    public static final ReluActivator INSTANCE = new ReluActivator();
    
    private ReluActivator() {} // Private constructor for singleton

    @Override
    public void activate(float[] input, float[] output) {
        checkLength(input, output);
        activateScalar(input, output);
    }

    @Override
    public void derivative(float[] input, float[] output) {
        checkLength(input, output);
        derivativeScalar(input, output);
    }

    private static void checkLength(float[] input, float[] output) {
        if (input.length != output.length) {
            throw new IllegalArgumentException(
                    "Input and output arrays must have same length: " +
                            "input=" + input.length + ", output=" + output.length
            );
        }
    }

    void activateVectorized(float[] input, float[] output) {
        activateScalar(input, output);
    }

    void activateScalar(float[] input, float[] output) {
        for (int i = 0; i < input.length; i++) {
            output[i] = input[i] > 0f ? input[i] : 0f;
        }
    }

    void derivativeVectorized(float[] input, float[] output) {
        derivativeScalar(input, output);
    }
    
    void derivativeScalar(float[] input, float[] output) {
        for (int i = 0; i < input.length; i++) {
            output[i] = input[i] > 0f ? 1f : 0f;
        }
    }
    
    @Override
    public void activate(float[] input, float[] output, ExecutorService executor) {
        if (Parallelization.shouldParallelize(input.length, executor)) {
            activateParallel(input, output, executor);
        } else {
            activate(input, output);
        }
    }
    
    @Override
    public void derivative(float[] input, float[] output, ExecutorService executor) {
        if (Parallelization.shouldParallelize(input.length, executor)) {
            derivativeParallel(input, output, executor);
        } else {
            derivative(input, output);
        }
    }
    
    private void activateParallel(float[] input, float[] output, ExecutorService executor) {
        checkLength(input, output);
        
        int numThreads = Parallelization.calculateOptimalThreads(input.length, executor);
        Parallelization.WorkRange[] ranges = Parallelization.splitWork(input.length, numThreads);
        
        Runnable[] tasks = new Runnable[ranges.length];
        for (int i = 0; i < ranges.length; i++) {
            final Parallelization.WorkRange range = ranges[i];
            tasks[i] = () -> {
                activateScalarRange(input, output, range.start, range.end);
            };
        }
        
        Parallelization.executeParallel(executor, tasks);
    }
    
    private void derivativeParallel(float[] input, float[] output, ExecutorService executor) {
        checkLength(input, output);
        
        int numThreads = Parallelization.calculateOptimalThreads(input.length, executor);
        Parallelization.WorkRange[] ranges = Parallelization.splitWork(input.length, numThreads);
        
        Runnable[] tasks = new Runnable[ranges.length];
        for (int i = 0; i < ranges.length; i++) {
            final Parallelization.WorkRange range = ranges[i];
            tasks[i] = () -> {
                derivativeScalarRange(input, output, range.start, range.end);
            };
        }
        
        Parallelization.executeParallel(executor, tasks);
    }
    
    private void activateVectorizedRange(float[] input, float[] output, int start, int end) {
        activateScalarRange(input, output, start, end);
    }
    
    private void activateScalarRange(float[] input, float[] output, int start, int end) {
        for (int i = start; i < end; i++) {
            output[i] = input[i] > 0f ? input[i] : 0f;
        }
    }
    
    private void derivativeVectorizedRange(float[] input, float[] output, int start, int end) {
        derivativeScalarRange(input, output, start, end);
    }
    
    private void derivativeScalarRange(float[] input, float[] output, int start, int end) {
        for (int i = start; i < end; i++) {
            output[i] = input[i] > 0f ? 1.0f : 0.0f;
        }
    }
}