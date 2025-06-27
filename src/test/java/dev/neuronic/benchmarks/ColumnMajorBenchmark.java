package dev.neuronic.benchmarks;

import dev.neuronic.net.math.NetMath;

import java.util.concurrent.ThreadLocalRandom;

/**
 * Benchmark comparing row-major vs column-major matrix layouts for pre-activations.
 */
public class ColumnMajorBenchmark {
    
    public static void main(String[] args) {
        System.out.println("=== Column-Major vs Row-Major Pre-Activations Benchmark ===");
        System.out.println();
        
        // Test realistic neural network layer sizes
        int[][] layerSizes = {
            {784, 128},    // MNIST -> hidden
            {128, 64},     // Hidden -> hidden
            {256, 256},    // Square layer
            {512, 1024},   // Expansion layer
            {2048, 512}    // Large -> smaller
        };
        
        for (int[] size : layerSizes) {
            int inputs = size[0];
            int neurons = size[1];
            
            System.out.println("=== Layer: " + inputs + " -> " + neurons + " ===");
            benchmarkPreActivations(inputs, neurons);
            System.out.println();
        }
    }
    
    private static void benchmarkPreActivations(int inputs, int neurons) {
        // Create test data
        float[] inputVector = createRandomArray(inputs);
        float[] biases = createRandomArray(neurons);
        
        // Row-major weights: weights[neuron][input] 
        float[][] rowMajorWeights = new float[neurons][inputs];
        fillRandomMatrix(rowMajorWeights);
        
        // Column-major weights: weights[input][neuron]
        float[][] columnMajorWeights = new float[inputs][neurons];
        fillRandomMatrix(columnMajorWeights);
        
        // Output buffers
        float[] rowMajorOutput = new float[neurons];
        float[] columnMajorOutput = new float[neurons];
        
        // Warmup  
        for (int i = 0; i < 10000; i++) {
            // Row-major computation (manual loop - old way)
            for (int j = 0; j < neurons; j++) {
                rowMajorOutput[j] = biases[j];
                for (int k = 0; k < inputs; k++) {
                    rowMajorOutput[j] += inputVector[k] * rowMajorWeights[j][k];
                }
            }
            NetMath.matrixPreActivationsColumnMajor(inputVector, columnMajorWeights, biases, columnMajorOutput);
        }
        
        // Benchmark row-major (old way - manual loops)
        long rowMajorTime = timeOperation(() -> {
            for (int j = 0; j < neurons; j++) {
                rowMajorOutput[j] = biases[j];
                for (int k = 0; k < inputs; k++) {
                    rowMajorOutput[j] += inputVector[k] * rowMajorWeights[j][k];
                }
            }
        }, 100000);
        
        // Benchmark column-major (new way)
        long columnMajorTime = timeOperation(() -> 
            NetMath.matrixPreActivationsColumnMajor(inputVector, columnMajorWeights, biases, columnMajorOutput), 100000);
        
        System.out.println("Pre-activations (" + inputs + "x" + neurons + "):");
        System.out.println("  Row-major:    " + rowMajorTime + " ns/op");
        System.out.println("  Column-major: " + columnMajorTime + " ns/op");
        if (columnMajorTime > 0) {
            System.out.println("  Speedup:      " + String.format("%.2fx", (double)rowMajorTime / columnMajorTime));
        }
    }
    
    private static float[] createRandomArray(int size) {
        ThreadLocalRandom random = ThreadLocalRandom.current();
        float[] array = new float[size];
        
        for (int i = 0; i < size; i++) {
            array[i] = (float) (random.nextGaussian() * 0.1);
        }
        return array;
    }
    
    private static void fillRandomMatrix(float[][] matrix) {
        ThreadLocalRandom random = ThreadLocalRandom.current();
        
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                matrix[i][j] = (float) (random.nextGaussian() * 0.1);
            }
        }
    }
    
    private static long timeOperation(Runnable operation, int iterations) {
        long startTime = System.nanoTime();
        
        for (int i = 0; i < iterations; i++) {
            operation.run();
        }
        
        long endTime = System.nanoTime();
        return (endTime - startTime) / iterations;
    }
}