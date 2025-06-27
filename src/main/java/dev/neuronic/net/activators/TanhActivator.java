package dev.neuronic.net.activators;

import dev.neuronic.net.math.Vectorization;
import dev.neuronic.net.math.Parallelization;

import java.util.concurrent.ExecutorService;

/**
 * Hyperbolic tangent activation function: f(x) = tanh(x) = (e^x - e^-x) / (e^x + e^-x)
 * 
 * Properties:
 * - Output range: (-1, 1)
 * - Zero-centered (unlike sigmoid)
 * - Derivative: f'(x) = 1 - tanh²(x)
 * 
 * Use for:
 * - Hidden layers in traditional networks
 * - RNNs and LSTMs (historically common)
 * - When you need zero-centered activations
 * 
 * Note: Less popular than ReLU for deep networks due to vanishing gradient issues.
 */
public final class TanhActivator implements Activator {
    
    public static final TanhActivator INSTANCE = new TanhActivator();
    
    private TanhActivator() {} // Singleton pattern
    
    @Override
    public void activate(float[] input, float[] output) {
        checkLength(input, output);
        
        if (Vectorization.shouldVectorize(input.length))
            activateVectorized(input, output);
        else
            activateScalar(input, output);
    }
    
    @Override
    public void derivative(float[] input, float[] output) {
        checkLength(input, output);
        
        if (Vectorization.shouldVectorize(input.length))
            derivativeVectorized(input, output);
        else
            derivativeScalar(input, output);
    }
    
    void activateVectorized(float[] input, float[] output) {
        // Delegate to scalar implementation for now
        activateScalar(input, output);
    }
    
    void activateScalar(float[] input, float[] output) {
        for (int i = 0; i < input.length; i++) {
            output[i] = (float) Math.tanh(input[i]);
        }
    }
    
    void derivativeVectorized(float[] tanhOutput, float[] output) {
        // Delegate to scalar implementation for now
        derivativeScalar(tanhOutput, output);
    }
    
    void derivativeScalar(float[] tanhOutput, float[] output) {
        for (int i = 0; i < tanhOutput.length; i++) {
            float tanh = tanhOutput[i];
            output[i] = 1.0f - tanh * tanh;
        }
    }
    
    @Override
    public void activate(float[] input, float[] output, ExecutorService executor) {
        if (Parallelization.shouldParallelize(input.length, executor))
            activateParallel(input, output, executor);
        else
            activate(input, output);
    }
    
    @Override
    public void derivative(float[] input, float[] output, ExecutorService executor) {
        if (Parallelization.shouldParallelize(input.length, executor))
            derivativeParallel(input, output, executor);
        else
            derivative(input, output);
    }
    
    private void activateParallel(float[] input, float[] output, ExecutorService executor) {
        checkLength(input, output);
        
        int numThreads = Parallelization.calculateOptimalThreads(input.length, executor);
        Parallelization.WorkRange[] ranges = Parallelization.splitWork(input.length, numThreads);
        
        Runnable[] tasks = new Runnable[ranges.length];
        for (int i = 0; i < ranges.length; i++) {
            final Parallelization.WorkRange range = ranges[i];
            tasks[i] = () -> {
                if (Vectorization.shouldVectorize(range.size))
                    activateVectorizedRange(input, output, range.start, range.end);
                else
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
                if (Vectorization.shouldVectorize(range.size))
                    derivativeVectorizedRange(input, output, range.start, range.end);
                else
                    derivativeScalarRange(input, output, range.start, range.end);
            };
        }
        
        Parallelization.executeParallel(executor, tasks);
    }
    
    private void activateVectorizedRange(float[] input, float[] output, int start, int end) {
        // Delegate to scalar implementation for now
        activateScalarRange(input, output, start, end);
    }
    
    private void activateScalarRange(float[] input, float[] output, int start, int end) {
        for (int i = start; i < end; i++) {
            output[i] = (float) Math.tanh(input[i]);
        }
    }
    
    private void derivativeVectorizedRange(float[] input, float[] output, int start, int end) {
        // Delegate to scalar implementation for now
        derivativeScalarRange(input, output, start, end);
    }
    
    private void derivativeScalarRange(float[] input, float[] output, int start, int end) {
        for (int i = start; i < end; i++) {
            float tanh = input[i];
            output[i] = 1.0f - tanh * tanh;
        }
    }
    
    private void checkLength(float[] input, float[] output) {
        if (input.length != output.length) {
            throw new IllegalArgumentException("Input and output arrays must have the same length");
        }
    }
}