package dev.neuronic.net.activators;

import dev.neuronic.net.math.Vectorization;
import dev.neuronic.net.math.Parallelization;

import java.util.concurrent.ExecutorService;

/**
 * Sigmoid activation function: f(x) = 1 / (1 + e^(-x))
 * 
 * Properties:
 * - Output range: (0, 1)
 * - S-shaped curve
 * - Derivative: f'(x) = f(x) * (1 - f(x))
 * 
 * Use for:
 * - Binary classification output layers (with binary cross-entropy)
 * - Gates in LSTM/GRU cells
 * - When you need probability-like outputs
 * 
 * Note: Suffers from vanishing gradients in deep networks, so ReLU is preferred for hidden layers.
 */
public final class SigmoidActivator implements Activator {
    
    public static final SigmoidActivator INSTANCE = new SigmoidActivator();
    
    private SigmoidActivator() {} // Singleton pattern
    
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
            output[i] = 1.0f / (1.0f + (float) Math.exp(-input[i]));
        }
    }
    
    void derivativeVectorized(float[] sigmoidOutput, float[] output) {
        // Delegate to scalar implementation for now
        derivativeScalar(sigmoidOutput, output);
    }
    
    void derivativeScalar(float[] sigmoidOutput, float[] output) {
        for (int i = 0; i < sigmoidOutput.length; i++) {
            float sigmoid = sigmoidOutput[i];
            output[i] = sigmoid * (1.0f - sigmoid);
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
            output[i] = 1.0f / (1.0f + (float) Math.exp(-input[i]));
        }
    }
    
    private void derivativeVectorizedRange(float[] input, float[] output, int start, int end) {
        // Delegate to scalar implementation for now
        derivativeScalarRange(input, output, start, end);
    }
    
    private void derivativeScalarRange(float[] input, float[] output, int start, int end) {
        for (int i = start; i < end; i++) {
            float sigmoid = input[i];
            output[i] = sigmoid * (1.0f - sigmoid);
        }
    }
    
    private void checkLength(float[] input, float[] output) {
        if (input.length != output.length) {
            throw new IllegalArgumentException("Input and output arrays must have the same length");
        }
    }
}